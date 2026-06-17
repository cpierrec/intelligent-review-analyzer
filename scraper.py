"""
scraper.py - Enterprise Async Web Veri Kazıma Katmanı
======================================================
Amaç: Hedef kaynaklardan ham verileri asyncio + httpx ile yüksek performanslı,
      dirençli ve kibarca (polite) bir şekilde çekmek.

Mimari Kararlar:
  - asyncio + httpx.AsyncClient: Python async I/O ile concurrent HTTP istekleri;
    GIL'den bağımsız gerçek paralelizm (I/O-bound workload için idealdir).
  - User-Agent Rotation Pool: 15 farklı tarayıcı UA; bot tespitini zorlaştırır.
  - ProxyRotator: Opsiyonel proxy havuzu; round-robin + health check ile rotation.
  - Jitter Sleep: İstekler arası rastgele bekleme; rate limiting + politeness.
  - Graceful Degradation: httpx kurulu değilse requests'e fallback; import error
    durumunda mock modda çalışmaya devam eder.
  - Sync Compat Wrapper: fetch_all_sync() → asyncio.run() ile mevcut main.py
    kodunu kırmadan kullanılabilir.

Enterprise Yenilikler (v2):
  - AsyncDataScraper : httpx.AsyncClient tabanlı tam async implementasyon
  - UserAgentPool    : 15 UA random rotation, seed'li determinizm opsiyonu
  - ProxyRotator     : Round-robin proxy seçimi, başarısız proxy'yi devre dışı bırakma
  - _jitter_sleep()  : base ± jitter uniform random sleep
  - fetch_all_sync() : asyncio.run() wrapper — backward compat

Kullanım:
    # Async (önerilen)
    scraper = AsyncDataScraper(network_cfg, data_cfg)
    reviews = await scraper.fetch_all()

    # Sync (mevcut main.py uyumluluğu)
    scraper = DataScraper(network_cfg, data_cfg)
    reviews = scraper.fetch_all()  # internally runs asyncio.run()
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin

# httpx opsiyonel bağımlılık — kurulu değilse requests'e fallback
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import DataSourceConfig, NetworkConfig
from models import RawReviewData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Özel İstisna Hiyerarşisi
# ---------------------------------------------------------------------------

class ScraperError(Exception):
    """Scraper katmanının temel hata sınıfı."""


class NetworkError(ScraperError):
    """Kurtarılamaz ağ hatası - tüm retry'lar tükendi."""
    def __init__(self, url: str, attempts: int, original_error: Exception) -> None:
        self.url = url
        self.attempts = attempts
        self.original_error = original_error
        super().__init__(
            f"Ag hatasi: {url} adresine {attempts} denemeden sonra ulasilamadi. "
            f"Hata: {type(original_error).__name__}: {original_error}"
        )


class ParseError(ScraperError):
    """HTML/JSON parse başarısız olduğunda fırlatılır."""


class RateLimitError(ScraperError):
    """HTTP 429 — Too Many Requests, backoff gerektirir."""
    def __init__(self, retry_after: Optional[int] = None) -> None:
        self.retry_after = retry_after or 60
        super().__init__(f"Rate limit asildi. {self.retry_after} saniye bekleniyor.")


# ---------------------------------------------------------------------------
# User-Agent Rotation Pool
# ---------------------------------------------------------------------------

class UserAgentPool:
    """
    15 farklı tarayıcı User-Agent barındıran rotation havuzu.

    Neden önemli?
      - Tek bir UA sürekli kullanılırsa bot olarak tespit edilir.
      - Gerçek tarayıcı UA'ları ile istekler daha doğal görünür.
      - Seed'li RNG ile determinizm sağlanabilir (test ortamları için).

    Kullanım:
        pool = UserAgentPool()
        ua = pool.get()  # Her çağrıda farklı UA döndürür
    """

    _USER_AGENTS: list[str] = [
        # Chrome - Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        # Chrome - macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        # Chrome - Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        # Firefox - Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0",
        # Firefox - macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) "
        "Gecko/20100101 Firefox/125.0",
        # Firefox - Linux
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0",
        # Safari - macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        # Safari - iOS
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
        # Edge - Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        # Edge - macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        # Chrome - Android
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
        # Opera - Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 OPR/110.0.0.0",
        # Brave - Windows (Brave sends standard Chrome UA)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        # Chrome - Windows (older version for variety)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # Firefox - Windows (older)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) "
        "Gecko/20100101 Firefox/122.0",
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        seed: None → gerçek rastgelelik; int → deterministik (test için).
        """
        self._rng = random.Random(seed)
        logger.debug(f"UserAgentPool oluşturuldu: {len(self._USER_AGENTS)} UA, seed={seed}")

    def get(self) -> str:
        """Havuzdan rastgele bir User-Agent döndür."""
        return self._rng.choice(self._USER_AGENTS)

    def get_headers(self) -> dict[str, str]:
        """Tam Accept/UA header dict'i döndür."""
        return {
            "User-Agent": self.get(),
            "Accept": "application/json, text/html, */*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "DNT": "1",
        }


# ---------------------------------------------------------------------------
# Proxy Rotation Altyapısı
# ---------------------------------------------------------------------------

class ProxyRotator:
    """
    Round-robin proxy rotation altyapısı.

    Özellikler:
      - Sırasıyla proxy'leri döndürür (round-robin)
      - Başarısız proxy'yi devre dışı bırakır (circuit breaker)
      - Tüm proxy'ler başarısız olursa proxy'siz (direct) çalışır (graceful degradation)
      - Proxy listesi boşsa direkt bağlantı kullanılır

    Kullanım:
        rotator = ProxyRotator(["http://proxy1:8080", "http://proxy2:8080"])
        proxy_dict = rotator.get_next()  # {"http": "...", "https": "..."}
    """

    def __init__(self, proxies: Optional[list[str]] = None) -> None:
        self._proxies: list[str] = proxies or []
        self._disabled: set[str] = set()
        self._index: int = 0
        if self._proxies:
            logger.info(f"ProxyRotator: {len(self._proxies)} proxy yüklendi.")
        else:
            logger.debug("ProxyRotator: Proxy listesi boş, direkt bağlantı kullanılacak.")

    @property
    def active_proxies(self) -> list[str]:
        """Devre dışı olmayan proxy listesi."""
        return [p for p in self._proxies if p not in self._disabled]

    def get_next(self) -> Optional[dict[str, str]]:
        """
        Sıradaki aktif proxy'yi döndür.
        Aktif proxy yoksa None döndür (direct connection).
        """
        active = self.active_proxies
        if not active:
            return None  # Direct connection — graceful degradation

        proxy_url = active[self._index % len(active)]
        self._index += 1
        return {"http": proxy_url, "https": proxy_url}

    def mark_failed(self, proxy_url: str) -> None:
        """Başarısız proxy'yi devre dışı bırak (circuit breaker)."""
        self._disabled.add(proxy_url)
        remaining = len(self.active_proxies)
        logger.warning(
            f"Proxy devre disi birakildi: {proxy_url} | "
            f"Kalan aktif proxy: {remaining}"
        )

    def reset(self) -> None:
        """Tüm devre dışı proxy'leri yeniden etkinleştir."""
        self._disabled.clear()
        self._index = 0
        logger.info("ProxyRotator sifirlandi: tum proxy'ler yeniden aktif.")


# ---------------------------------------------------------------------------
# Exponential Backoff Yardımcı Sınıfı
# ---------------------------------------------------------------------------

class ExponentialBackoff:
    """
    Jitter'lı exponential backoff algoritması.

    Formül: min(max_delay, base * 2^attempt) + uniform(0, jitter_factor * capped)
    """

    def __init__(
        self,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter_factor: float = 0.25,
    ) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

    def get_delay(self, attempt: int) -> float:
        """attempt numarasına göre bekleme süresini hesapla (saniye)."""
        exponential = self.base_delay * (2 ** attempt)
        capped = min(exponential, self.max_delay)
        jitter = capped * self.jitter_factor * random.random()
        return capped + jitter

    def sleep(self, attempt: int) -> None:
        """Hesaplanan süre kadar bekle ve logla."""
        delay = self.get_delay(attempt)
        logger.info(f"Retry bekleniyor: {delay:.1f}s (deneme #{attempt + 1})")
        time.sleep(delay)

    async def async_sleep(self, attempt: int) -> None:
        """Async versiyonu — event loop'u bloklamaz."""
        delay = self.get_delay(attempt)
        logger.info(f"Async retry bekleniyor: {delay:.1f}s (deneme #{attempt + 1})")
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Jitter Sleep Yardımcı Fonksiyonu
# ---------------------------------------------------------------------------

def _jitter_sleep(base: float = 1.0, jitter: float = 0.5) -> None:
    """
    Kibarca tarama için base ± jitter uniform random sleep.
    Örn: base=1.0, jitter=0.5 → [0.5, 1.5] arası rastgele bekleme.
    """
    actual = base + random.uniform(-jitter, jitter)
    actual = max(0.1, actual)  # Minimum 100ms
    time.sleep(actual)


async def _async_jitter_sleep(base: float = 1.0, jitter: float = 0.5) -> None:
    """Async versiyonu — event loop'u bloklamaz."""
    actual = base + random.uniform(-jitter, jitter)
    actual = max(0.05, actual)
    await asyncio.sleep(actual)


# ---------------------------------------------------------------------------
# Sync HTTP Session Fabrikası (requests — fallback)
# ---------------------------------------------------------------------------

def build_resilient_session(
    network_cfg: NetworkConfig,
    ua_pool: Optional[UserAgentPool] = None,
) -> requests.Session:
    """
    urllib3 retry politikası ile donatılmış requests.Session üretir.
    ua_pool varsa her istekte farklı UA kullanılır.
    """
    retry_policy = Retry(
        total=network_cfg.max_retries,
        backoff_factor=network_cfg.backoff_factor,
        status_forcelist=network_cfg.retry_status_codes,
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=network_cfg.max_concurrent_requests,
        pool_maxsize=network_cfg.max_concurrent_requests * 2,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = ua_pool.get_headers() if ua_pool else {
        "User-Agent": network_cfg.user_agent,
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    session.headers.update(headers)
    return session


# ---------------------------------------------------------------------------
# Async Data Scraper (httpx tabanlı — Enterprise Ana Implementasyon)
# ---------------------------------------------------------------------------

class AsyncDataScraper:
    """
    httpx.AsyncClient tabanlı enterprise async web scraper.

    httpx neden requests'ten üstün?
      - Native async/await support — event loop'u bloklamaz
      - HTTP/2 desteği (requests HTTP/1.1 only)
      - Daha iyi timeout kontrolü (connect + read ayrı)
      - Modern Python type hints ile uyumlu

    Özellikler:
      - UserAgentPool ile her istekte farklı UA
      - ProxyRotator ile opsiyonel proxy rotation
      - Jitter'lı async sleep (polite crawling)
      - Concurrent fetch (asyncio.gather) — paralel istek desteği
      - Graceful degradation: httpx yoksa ScraperError fırlatır
    """

    def __init__(
        self,
        network_cfg: NetworkConfig,
        data_cfg: DataSourceConfig,
        proxies: Optional[list[str]] = None,
        ua_seed: Optional[int] = None,
    ) -> None:
        if not _HTTPX_AVAILABLE:
            raise ScraperError(
                "httpx kurulu degil. Kurmak icin: pip install httpx\n"
                "Alternatif: DataScraper kullanin (requests tabanli)."
            )
        self.network_cfg = network_cfg
        self.data_cfg = data_cfg
        self.ua_pool = UserAgentPool(seed=ua_seed)
        self.proxy_rotator = ProxyRotator(proxies or [])
        self.backoff = ExponentialBackoff(
            base_delay=network_cfg.backoff_factor,
            max_delay=30.0,
        )
        self._request_count = 0
        logger.info(
            f"AsyncDataScraper baslatildi (httpx). "
            f"Kaynak: {data_cfg.source_type} | Hedef: {data_cfg.max_records} kayit"
        )

    def _build_httpx_client(self) -> "httpx.AsyncClient":
        """Her fetch session'ı için yeni bir httpx.AsyncClient oluştur."""
        proxy = self.proxy_rotator.get_next()
        headers = self.ua_pool.get_headers()

        kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": httpx.Timeout(
                connect=10.0,
                read=float(self.network_cfg.request_timeout),
                write=10.0,
                pool=5.0,
            ),
            "follow_redirects": True,
            "http2": False,  # HTTP/2 opsiyonel, bazı API'lerle uyumsuzluk olabilir
        }
        if proxy:
            kwargs["proxies"] = proxy

        return httpx.AsyncClient(**kwargs)

    async def _async_request(
        self,
        url: str,
        params: Optional[dict] = None,
        client: Optional["httpx.AsyncClient"] = None,
    ) -> "httpx.Response":
        """
        Tek bir async HTTP GET isteği — retry + backoff + UA rotation dahil.

        Her retry'da yeni User-Agent kullanılır (daha gerçekçi tarama).
        """
        self._request_count += 1
        logger.debug(f"Async HTTP GET #{self._request_count}: {url}")

        # Kibarca bekleme (ilk istekte skip)
        if self._request_count > 1:
            await _async_jitter_sleep(
                base=self.network_cfg.request_delay,
                jitter=self.network_cfg.request_delay * 0.5,
            )

        last_exception: Optional[Exception] = None
        _own_client = client is None

        if _own_client:
            client = self._build_httpx_client()

        try:
            for attempt in range(self.network_cfg.max_retries):
                # Her retry'da yeni UA header ekle
                headers = self.ua_pool.get_headers()

                try:
                    response = await client.get(url, params=params, headers=headers)

                    # 429: Rate limit
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(
                            f"Rate limit (429). {retry_after}s bekleniyor. URL: {url}"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    # 5xx: Sunucu hatası
                    if response.status_code >= 500:
                        logger.warning(
                            f"Sunucu hatasi {response.status_code} "
                            f"(deneme {attempt + 1}/{self.network_cfg.max_retries}). "
                            f"URL: {url}"
                        )
                        await self.backoff.async_sleep(attempt)
                        continue

                    # 4xx (429 hariç): retry'a değmez
                    if 400 <= response.status_code < 500:
                        logger.error(
                            f"Client hatasi {response.status_code} (retry yok). "
                            f"URL: {url}"
                        )
                        raise ScraperError(
                            f"HTTP {response.status_code} hatasi: {url}"
                        )

                    response.raise_for_status()
                    logger.debug(
                        f"Basarili yanit: status={response.status_code}, "
                        f"size={len(response.content)} bytes"
                    )
                    return response

                except httpx.ConnectError as e:
                    logger.warning(f"Baglanti hatasi (deneme {attempt + 1}): {e}")
                    last_exception = e
                    await self.backoff.async_sleep(attempt)

                except httpx.TimeoutException as e:
                    logger.warning(f"Timeout (deneme {attempt + 1}): {e}")
                    last_exception = e
                    await self.backoff.async_sleep(attempt)

                except httpx.RequestError as e:
                    logger.warning(f"Istek hatasi (deneme {attempt + 1}): {e}")
                    last_exception = e
                    await self.backoff.async_sleep(attempt)

        finally:
            if _own_client and client:
                await client.aclose()

        raise NetworkError(
            url=url,
            attempts=self.network_cfg.max_retries,
            original_error=last_exception or Exception("Bilinmeyen hata"),
        )

    async def _parse_json_response(self, response: "httpx.Response") -> Any:
        """JSON yanıtını güvenli biçimde ayrıştırır."""
        try:
            return response.json()
        except Exception as e:
            snippet = response.text[:200] if response.text else "(bos)"
            raise ParseError(
                f"JSON parse hatasi: {e}. Yanit baslangici: {snippet!r}"
            ) from e

    def _parse_html_content(
        self,
        content: bytes,
        parser: str = "html.parser",
    ) -> BeautifulSoup:
        """HTML bytes içeriğini BeautifulSoup ile ayrıştırır."""
        try:
            return BeautifulSoup(content, parser)
        except Exception as e:
            raise ParseError(f"HTML parse hatasi: {e}") from e

    async def fetch_from_api_async(self, limit: int) -> list[dict]:
        """
        JSONPlaceholder API'sinden async olarak post + comment verisi çeker.

        asyncio.gather ile paralel comment fetch — N kat daha hızlı.
        """
        logger.info(f"Async API'den veri cekiliyor: limit={limit}")
        base_url = self.data_cfg.mock_api_base_url
        raw_records: list[dict] = []

        async with self._build_httpx_client() as client:
            # Adım 1: Post listesini çek
            posts_url = f"{base_url.rstrip('/')}/posts"
            try:
                resp = await self._async_request(
                    posts_url,
                    params={"_limit": min(limit, 100)},
                    client=client,
                )
                posts = await self._parse_json_response(resp)
                logger.info(f"  {len(posts)} post cekildi")
            except (NetworkError, ParseError) as e:
                logger.error(f"Post verisi cekilemedi: {e}")
                raise

            # Adım 2: Her post için comment'leri paralel çek
            async def fetch_comments(post: dict) -> list[dict]:
                post_id = post.get("id")
                if not post_id:
                    return []
                comments_url = f"{base_url.rstrip('/')}/posts/{post_id}/comments"
                try:
                    r = await self._async_request(comments_url, client=client)
                    comments = await self._parse_json_response(r)
                    enriched = []
                    for comment in comments[:2]:
                        enriched.append({
                            "post_id": post_id,
                            "post_title": post.get("title", ""),
                            "post_body": post.get("body", ""),
                            "comment_id": comment.get("id"),
                            "comment_name": comment.get("name", ""),
                            "comment_email": comment.get("email", ""),
                            "comment_body": comment.get("body", ""),
                            "source_url": comments_url,
                        })
                    return enriched
                except NetworkError as e:
                    logger.warning(f"Post {post_id} yorumlari alinamadi: {e}")
                    return []

            # Paralel fetch — tüm postları aynı anda işle
            target_posts = posts[:min(limit, len(posts))]
            results = await asyncio.gather(
                *[fetch_comments(p) for p in target_posts],
                return_exceptions=False,
            )
            for batch in results:
                raw_records.extend(batch)
                if len(raw_records) >= limit:
                    break

        logger.info(f"  Async API: {len(raw_records)} ham kayit toplandi")
        return raw_records[:limit]

    async def fetch_from_html_async(self, url: str) -> list[dict]:
        """
        Statik HTML sayfasından async olarak BeautifulSoup ile veri çıkarır.
        """
        logger.info(f"Async HTML kaziniyor: {url}")
        raw_records: list[dict] = []

        try:
            resp = await self._async_request(url)
            soup = self._parse_html_content(resp.content)
        except (NetworkError, ParseError) as e:
            logger.error(f"HTML sayfasi cekilemedi: {e}")
            raise

        product_pods = soup.select("article.product_pod")
        if not product_pods:
            logger.warning("HTML'de urun karti bulunamadi.")
            return raw_records

        rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

        for idx, pod in enumerate(product_pods[:self.data_cfg.max_records]):
            try:
                title_tag = pod.select_one("h3 > a")
                title = (
                    title_tag.get("title") or title_tag.text.strip()
                    if title_tag else "Unknown"
                )
                price_tag = pod.select_one("p.price_color")
                price_text = price_tag.text.strip() if price_tag else "N/A"

                rating_tag = pod.select_one("p.star-rating")
                rating_class = rating_tag.get("class", []) if rating_tag else []
                rating_word = next(
                    (w for w in rating_class if w in rating_map), "Zero"
                )
                star_rating = rating_map.get(rating_word, 0)

                availability_tag = pod.select_one("p.availability")
                availability = (
                    availability_tag.text.strip() if availability_tag else "Unknown"
                )

                link_tag = pod.select_one("h3 > a")
                detail_href = link_tag.get("href", "") if link_tag else ""
                detail_url = urljoin(url, detail_href)

                raw_records.append({
                    "post_id": f"html_{idx + 1}",
                    "post_title": title,
                    "post_body": f"{title} - Price: {price_text} - Rating: {star_rating}/5",
                    "comment_id": idx + 1,
                    "comment_name": f"Product Review #{idx + 1}",
                    "comment_email": "scraper@system.local",
                    "comment_body": (
                        f"Product: {title}. Price: {price_text}. "
                        f"Customer rating: {star_rating} out of 5 stars. "
                        f"Availability: {availability}. "
                        f"Customers who rated this "
                        f"{'highly recommend' if star_rating >= 4 else 'have mixed opinions about'}"
                        f" this product."
                    ),
                    "source_url": detail_url,
                    "star_rating": star_rating,
                    "price": price_text,
                    "availability": availability,
                })
            except Exception as e:
                logger.warning(f"Urun karti #{idx} ayristirilamamadi: {e}")
                continue

        logger.info(f"  Async HTML: {len(raw_records)} kayit cikarildi")
        return raw_records

    def _normalize_to_model(
        self, raw_record: dict, index: int
    ) -> Optional[RawReviewData]:
        """Ham dict'i RawReviewData modeline normalize et."""
        try:
            text_parts = []
            if raw_record.get("post_title"):
                text_parts.append(f"Title: {raw_record['post_title']}")
            if raw_record.get("comment_name"):
                text_parts.append(f"Review Title: {raw_record['comment_name']}")
            if raw_record.get("comment_body"):
                text_parts.append(f"Review: {raw_record['comment_body']}")
            if raw_record.get("post_body") and not raw_record.get("comment_body"):
                text_parts.append(f"Content: {raw_record['post_body']}")

            combined_text = "\n".join(text_parts).strip()
            if not combined_text:
                return None

            source_id = str(
                raw_record.get("comment_id")
                or raw_record.get("post_id")
                or f"rec_{index}"
            )

            return RawReviewData(
                source_id=source_id,
                source_url=raw_record.get("source_url"),
                raw_text=combined_text,
                metadata={
                    k: v for k, v in raw_record.items()
                    if k not in ("comment_body", "post_body", "comment_name")
                    and v is not None
                },
            )
        except Exception as e:
            logger.warning(f"Normalizasyon hatasi (kayit #{index}): {e}")
            return None

    async def fetch_all(self) -> list[RawReviewData]:
        """
        Yapılandırılan kaynak tipine göre async veri çekme ve normalizasyon.
        """
        logger.info(
            f"=== Async Veri Cekme Basliyor === "
            f"Kaynak: {self.data_cfg.source_type} | "
            f"Hedef: {self.data_cfg.max_records} kayit"
        )
        start_time = time.time()
        raw_dicts: list[dict] = []

        try:
            if self.data_cfg.source_type == "mock_api":
                raw_dicts = await self.fetch_from_api_async(self.data_cfg.max_records)
            elif self.data_cfg.source_type == "html_scrape":
                raw_dicts = await self.fetch_from_html_async(
                    "https://books.toscrape.com/catalogue/page-1.html"
                )
            else:
                raise ScraperError(
                    f"Desteklenmeyen kaynak tipi: {self.data_cfg.source_type}"
                )
        except ScraperError:
            raise
        except Exception as e:
            raise ScraperError(f"Async veri cekme hatasi: {e}") from e

        normalized: list[RawReviewData] = []
        for idx, raw in enumerate(raw_dicts):
            model = self._normalize_to_model(raw, idx)
            if model:
                normalized.append(model)

        elapsed = time.time() - start_time
        logger.info(
            f"=== Async Veri Cekme Tamamlandi === "
            f"Ham: {len(raw_dicts)} | Normalize: {len(normalized)} | "
            f"Sure: {elapsed:.2f}s | Istek: {self._request_count}"
        )
        return normalized


# ---------------------------------------------------------------------------
# Sync Compat Wrapper — Mevcut main.py uyumluluğu için
# ---------------------------------------------------------------------------

class DataScraper:
    """
    Backward-compatible sync wrapper.

    Dahili olarak AsyncDataScraper veya eski requests tabanlı implementasyonu
    kullanır. main.py'deki `scraper.fetch_all()` çağrısı değişmeden çalışır.

    Seçim mantığı:
      1. httpx kuruluysa → AsyncDataScraper(asyncio.run ile wrap)
      2. httpx yoksa → Eski requests tabanlı implementasyon (graceful degradation)
    """

    def __init__(
        self,
        network_cfg: NetworkConfig,
        data_cfg: DataSourceConfig,
        proxies: Optional[list[str]] = None,
        ua_seed: Optional[int] = None,
    ) -> None:
        self.network_cfg = network_cfg
        self.data_cfg = data_cfg
        self.ua_pool = UserAgentPool(seed=ua_seed)
        self.proxy_rotator = ProxyRotator(proxies or [])
        self.backoff = ExponentialBackoff(base_delay=network_cfg.backoff_factor)
        self._request_count = 0

        # httpx varsa async implementasyonu kullan
        if _HTTPX_AVAILABLE:
            self._async_scraper = AsyncDataScraper(
                network_cfg, data_cfg, proxies=proxies, ua_seed=ua_seed
            )
            self._use_async = True
        else:
            self._async_scraper = None
            self._use_async = False
            # Fallback: requests session
            self.session = build_resilient_session(network_cfg, self.ua_pool)

        logger.info(
            f"DataScraper baslatildi. "
            f"Backend: {'httpx/async' if self._use_async else 'requests/sync'} | "
            f"Kaynak: {data_cfg.source_type}"
        )

    # --- Sync fallback metodları (httpx olmadığında) ---

    def _make_request(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> requests.Response:
        """Sync HTTP GET — requests session ile."""
        if self._request_count > 0:
            _jitter_sleep(
                base=self.network_cfg.request_delay,
                jitter=self.network_cfg.request_delay * 0.5,
            )
        self._request_count += 1
        # Her istekte yeni UA header'ı güncelle
        self.session.headers.update({"User-Agent": self.ua_pool.get()})

        # Opsiyonel proxy rotation
        proxy_dict = self.proxy_rotator.get_next()

        last_exc: Optional[Exception] = None
        for attempt in range(self.network_cfg.max_retries):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    timeout=self.network_cfg.request_timeout,
                    proxies=proxy_dict,
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limit 429. {retry_after}s bekleniyor.")
                    time.sleep(retry_after)
                    continue
                if resp.status_code >= 500:
                    logger.warning(f"Sunucu hatasi {resp.status_code}, retry...")
                    self.backoff.sleep(attempt)
                    continue
                resp.raise_for_status()
                return resp
            except requests.exceptions.RequestException as e:
                last_exc = e
                self.backoff.sleep(attempt)

        raise NetworkError(
            url=url,
            attempts=self.network_cfg.max_retries,
            original_error=last_exc or Exception("Bilinmeyen hata"),
        )

    def _fetch_from_api_sync(self, limit: int) -> list[dict]:
        """requests tabanlı sync API fetch (httpx yoksa kullanılır)."""
        base_url = self.data_cfg.mock_api_base_url
        raw_records: list[dict] = []

        posts_url = f"{base_url.rstrip('/')}/posts"
        resp = self._make_request(posts_url, params={"_limit": min(limit, 100)})
        posts = resp.json()

        for post in posts[:min(limit, len(posts))]:
            post_id = post.get("id")
            if not post_id:
                continue
            comments_url = f"{base_url.rstrip('/')}/posts/{post_id}/comments"
            try:
                r = self._make_request(comments_url)
                comments = r.json()
                for comment in comments[:2]:
                    raw_records.append({
                        "post_id": post_id,
                        "post_title": post.get("title", ""),
                        "post_body": post.get("body", ""),
                        "comment_id": comment.get("id"),
                        "comment_name": comment.get("name", ""),
                        "comment_email": comment.get("email", ""),
                        "comment_body": comment.get("body", ""),
                        "source_url": comments_url,
                    })
                if len(raw_records) >= limit:
                    break
            except NetworkError as e:
                logger.warning(f"Post {post_id} atlandi: {e}")
                continue

        return raw_records[:limit]

    def _normalize_to_model(
        self, raw_record: dict, index: int
    ) -> Optional[RawReviewData]:
        """Ham dict'i RawReviewData modeline normalize et."""
        try:
            text_parts = []
            if raw_record.get("post_title"):
                text_parts.append(f"Title: {raw_record['post_title']}")
            if raw_record.get("comment_name"):
                text_parts.append(f"Review Title: {raw_record['comment_name']}")
            if raw_record.get("comment_body"):
                text_parts.append(f"Review: {raw_record['comment_body']}")
            if raw_record.get("post_body") and not raw_record.get("comment_body"):
                text_parts.append(f"Content: {raw_record['post_body']}")

            combined_text = "\n".join(text_parts).strip()
            if not combined_text:
                return None

            source_id = str(
                raw_record.get("comment_id")
                or raw_record.get("post_id")
                or f"rec_{index}"
            )

            return RawReviewData(
                source_id=source_id,
                source_url=raw_record.get("source_url"),
                raw_text=combined_text,
                metadata={
                    k: v for k, v in raw_record.items()
                    if k not in ("comment_body", "post_body", "comment_name")
                    and v is not None
                },
            )
        except Exception as e:
            logger.warning(f"Normalizasyon hatasi (kayit #{index}): {e}")
            return None

    def fetch_all(self) -> list[RawReviewData]:
        """
        Ana veri çekme metodu — sync interface.

        httpx varsa async implementasyonu asyncio.run() ile çalıştırır.
        httpx yoksa requests tabanlı sync implementasyonu kullanır.
        """
        logger.info(
            f"=== Veri Cekme Basliyor === "
            f"Backend: {'httpx' if self._use_async else 'requests'} | "
            f"Kaynak: {self.data_cfg.source_type}"
        )
        start_time = time.time()

        if self._use_async and self._async_scraper:
            # asyncio.run() — yeni event loop'ta async kodu çalıştır
            try:
                normalized = asyncio.run(self._async_scraper.fetch_all())
            except RuntimeError:
                # Halihazırda çalışan event loop varsa (Jupyter, vb.)
                import nest_asyncio  # type: ignore
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                normalized = loop.run_until_complete(self._async_scraper.fetch_all())
            elapsed = time.time() - start_time
            logger.info(
                f"=== Veri Cekme Tamamlandi (async) === "
                f"Normalize: {len(normalized)} | Sure: {elapsed:.2f}s"
            )
            return normalized

        # Sync fallback
        raw_dicts: list[dict] = []
        try:
            if self.data_cfg.source_type == "mock_api":
                raw_dicts = self._fetch_from_api_sync(self.data_cfg.max_records)
            else:
                raise ScraperError(
                    f"Sync modda desteklenmeyen kaynak: {self.data_cfg.source_type}"
                )
        except ScraperError:
            raise
        except Exception as e:
            raise ScraperError(f"Sync veri cekme hatasi: {e}") from e

        normalized = []
        for idx, raw in enumerate(raw_dicts):
            model = self._normalize_to_model(raw, idx)
            if model:
                normalized.append(model)

        elapsed = time.time() - start_time
        logger.info(
            f"=== Veri Cekme Tamamlandi (sync/requests) === "
            f"Ham: {len(raw_dicts)} | Normalize: {len(normalized)} | "
            f"Sure: {elapsed:.2f}s"
        )
        return normalized

