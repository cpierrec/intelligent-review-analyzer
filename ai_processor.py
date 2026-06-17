"""
ai_processor.py - Yapay Zeka Entegrasyon Katmanı (Enterprise Edition)
======================================================================
Amaç: Ham, yapılandırılmamış metinleri LLM API'leri aracılığıyla
      katı JSON şemasına uyan iş zekasına dönüştürmek.

Enterprise Geliştirmeleri:
  1. tiktoken entegrasyonu — API'ye gitmeden önce token sayısı hesaplanır;
     batch boyutu dinamik olarak küçülür, context window taşması önlenir.
  2. Dinamik batch splitter — max_tokens_per_batch konfigürasyonuna göre
     kayıtları otomatik böler; maliyet ve hız dengelenir.
  3. Akıllı MockAIProvider — gelen metindeki negatif anahtar kelimelere
     (broken, damaged, terrible, kötü, bozuk vb.) duyarlı; tutarlı
     negatif sentiment + HIGH urgency üretir. Seed'li deterministik çıktı.
  4. Exponential backoff with jitter — tenacity yerine manual implementasyon
     (dış bağımlılık minimulaştırma); her provider'da ayrı retry stratejisi.
  5. OpenAI Structured Outputs — response_format=json_schema ile model
     constraint'lı JSON üretir; parse hatası sıfıra iner.

Mimari: Strategy Pattern — BaseAIProvider sözleşmesi değiştirmeden
        yeni provider eklenebilir (Open/Closed Principle).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from config import AIConfig
from models import (
    AIAnalyzedReview,
    AIBatchResult,
    DataQuality,
    ProductCategory,
    RawReviewData,
    SentimentLabel,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Özel İstisna Hiyerarşisi
# ---------------------------------------------------------------------------

class AIProcessorError(Exception):
    """AI işleme katmanının temel hata sınıfı."""
    pass


class AIAPIError(AIProcessorError):
    """LLM API çağrısı başarısız olduğunda fırlatılır."""
    def __init__(self, provider: str, status_code: Optional[int], message: str) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] API hatası (HTTP {status_code}): {message}")


class AIResponseParseError(AIProcessorError):
    """LLM yanıtı beklenen JSON şemasına uymadığında fırlatılır."""
    pass


class AIRateLimitError(AIProcessorError):
    """API rate limit aşıldığında fırlatılır."""
    def __init__(self, provider: str, retry_after: int = 60) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"[{provider}] Rate limit aşıldı. {retry_after}s sonra tekrar dene."
        )


class TokenBudgetExceededError(AIProcessorError):
    """
    Tek bir kayıt bile max_tokens_per_request sınırını geçiyorsa fırlatılır.
    Bu kayıt atlanır; pipeline durdurmaz.
    """
    def __init__(self, source_id: str, token_count: int, limit: int) -> None:
        self.source_id = source_id
        self.token_count = token_count
        self.limit = limit
        super().__init__(
            f"Kayıt '{source_id}' token limitini aşıyor: "
            f"{token_count} > {limit}. Kayıt atlanıyor."
        )


# ---------------------------------------------------------------------------
# Token Sayıcı (tiktoken entegrasyonu)
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    tiktoken kullanarak LLM API'ye gönderilecek metinlerin
    token sayısını hesaplar.

    Neden önemli?
      - GPT-4o-mini context window: 128k token
      - Batch çok büyükse API 400 hatası döner
      - Token başına maliyet hesabı için gerçek sayı gerekir

    Fallback: tiktoken import başarısız olursa (Python 3.14 uyumsuzluk),
    word-count tabanlı heuristic kullanılır (~1.3 token/kelime).
    """

    # Model → encoding adı eşlemesi
    _ENCODING_MAP: dict[str, str] = {
        "gpt-4o":          "o200k_base",
        "gpt-4o-mini":     "o200k_base",
        "gpt-4":           "cl100k_base",
        "gpt-3.5-turbo":   "cl100k_base",
        "claude-3-5-haiku-20241022": "cl100k_base",  # Claude için yaklaşık
        "mock-v1.0-deterministic":   "cl100k_base",
    }
    # Anthropic modelleri tiktoken desteklemez; kelime bazlı heuristic kullan
    _ANTHROPIC_MODELS = frozenset({
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    })

    def __init__(self, model_name: str) -> None:
        self._model = model_name
        self._encoder = None
        self._use_heuristic = False

        if model_name in self._ANTHROPIC_MODELS:
            self._use_heuristic = True
            logger.debug(f"TokenCounter: Anthropic modeli, heuristic mod aktif.")
            return

        encoding_name = self._ENCODING_MAP.get(model_name, "cl100k_base")
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(encoding_name)
            logger.debug(
                f"TokenCounter başlatıldı: model={model_name}, "
                f"encoding={encoding_name}"
            )
        except ImportError:
            logger.warning(
                "tiktoken bulunamadı. Heuristic token sayıcı kullanılıyor. "
                "Tam doğruluk için: pip install tiktoken"
            )
            self._use_heuristic = True
        except Exception as e:
            logger.warning(f"tiktoken yüklenemedi ({e}), heuristic mod aktif.")
            self._use_heuristic = True

    def count(self, text: str) -> int:
        """
        Verilen metnin token sayısını döndürür.
        tiktoken mevcut değilse kelime bazlı heuristic kullanır.
        """
        if not text:
            return 0
        if self._use_heuristic or self._encoder is None:
            # Heuristic: ~1.3 token per word (İngilizce ortalama)
            word_count = len(text.split())
            return math.ceil(word_count * 1.3)
        return len(self._encoder.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        OpenAI mesaj listesinin toplam token sayısını hesaplar.
        Her mesaj için ~4 token overhead (role, formatting) eklenir.
        """
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += 4  # role overhead
        total += 2  # conversation overhead
        return total

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """
        Tahmini API maliyeti (USD).
        Fiyatlar Haziran 2024 itibariyle; production'da güncel tutulmalı.
        """
        pricing: dict[str, dict[str, float]] = {
            "gpt-4o-mini":     {"input": 0.00015, "output": 0.00060},  # per 1K token
            "gpt-4o":          {"input": 0.00500, "output": 0.01500},
            "gpt-4":           {"input": 0.03000, "output": 0.06000},
            "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.00300, "output": 0.01500},
        }
        rates = pricing.get(self._model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000
        return round(cost, 6)


# ---------------------------------------------------------------------------
# Prompt Mühendisliği
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior e-commerce data analyst AI with expertise in
customer sentiment analysis and business intelligence extraction.

Your task is to analyze customer product reviews and extract structured business intelligence.

CRITICAL RULES — MUST FOLLOW EXACTLY:
1. Respond with ONLY valid JSON. No markdown, no explanation, no code blocks.
2. Your JSON MUST match the provided schema exactly — all required fields must be present.
3. Use null (not "null") for optional fields when information is unavailable.
4. sentiment_score: float between -1.0 (very negative) and +1.0 (very positive).
5. confidence_score: float between 0.0 and 1.0 — how confident you are in your analysis.
6. reviewer_rating: null if no explicit rating found; otherwise float 1.0-5.0.
7. reviewer_expertise: ONLY "novice", "intermediate", or "expert" — nothing else.
8. For urgency_level: "critical" > "high" > "medium" > "low".
9. Be data-driven and objective. Base ALL scores on actual review content.
10. If the review is in a non-English language, still respond in English JSON."""


def build_analysis_prompt(raw_text: str, json_schema: str) -> str:
    """
    LLM'e gönderilecek kullanıcı prompt'unu oluşturur.
    Schema direkt gömülü → LLM'in şemadan sapması engellenir.
    """
    return (
        f"Analyze the following customer review and extract structured data.\n\n"
        f"REVIEW TEXT:\n---\n{raw_text}\n---\n\n"
        f"Required JSON output schema:\n{json_schema}\n\n"
        f"Respond with ONLY the JSON object. No other text."
    )


def build_batch_prompt(reviews: list[dict[str, str]], json_schema: str) -> str:
    """
    N yorumu tek API çağrısında işlemek için batch prompt.
    Her yorum source_id ile etiketlenir; eşleştirme hatasını önler.
    """
    numbered = "\n\n".join(
        f"REVIEW #{i + 1} [source_id: {r['source_id']}]:\n{r['text']}"
        for i, r in enumerate(reviews)
    )
    return (
        f"Analyze each of the {len(reviews)} customer reviews below independently.\n\n"
        f"{numbered}\n\n"
        f"Output: JSON array with {len(reviews)} objects, each matching this schema:\n"
        f"{json_schema}\n\n"
        f"IMPORTANT: Preserve the exact source_id from each review. "
        f"Respond with ONLY the JSON array."
    )


# ---------------------------------------------------------------------------
# Soyut AI Provider Arayüzü (Strategy Pattern)
# ---------------------------------------------------------------------------

class BaseAIProvider(ABC):
    """
    Tüm AI sağlayıcıların implemente etmesi gereken sözleşme.
    Strategy pattern: main.py kodu değiştirmeden provider swap yapılabilir.
    """

    @abstractmethod
    def analyze_single(self, raw_text: str, source_id: str) -> dict[str, Any]:
        """Tek metni analiz et ve ham dict döndür."""
        ...

    @abstractmethod
    def analyze_batch(self, reviews: list[RawReviewData]) -> list[AIAnalyzedReview]:
        """
        RawReviewData listesini analiz eder.
        Her provider kendi iç formatına dönüştürür ama dışa
        her zaman validate edilmiş AIAnalyzedReview listesi döner.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Kullanılan model adı."""
        ...

    @property
    def supports_batch(self) -> bool:
        """Bu provider batch işlemeyi destekliyor mu?"""
        return True


# ---------------------------------------------------------------------------
# Exponential Backoff Yardımcı Fonksiyonu
# ---------------------------------------------------------------------------

def _backoff_sleep(attempt: int, base: float = 2.0, cap: float = 60.0) -> None:
    """
    Jitter'lı exponential backoff.
    Formül: min(cap, base^attempt) + uniform(0, 1)
    Thundering herd önlemek için random jitter eklenir.
    """
    delay = min(cap, base ** attempt) + random.uniform(0.0, 1.0)
    logger.info(f"  Backoff: {delay:.1f}s bekleniyor (deneme #{attempt + 1})...")
    time.sleep(delay)


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseAIProvider):
    """
    OpenAI GPT modelleri için AI sağlayıcı.
    openai>=1.30.0 SDK kullanır.

    Structured Outputs: response_format={"type": "json_schema", "json_schema": {...}}
    ile model JSON'u şemaya uygun üretmeye constraint edilir.
    Bu yaklaşım json_object moduna göre daha güvenilirdir.
    """

    def __init__(self, config: AIConfig) -> None:
        try:
            from openai import OpenAI
            from openai import RateLimitError, APIStatusError, APIConnectionError
            self._RateLimitError = RateLimitError
            self._APIStatusError = APIStatusError
            self._APIConnectionError = APIConnectionError
        except ImportError as e:
            raise AIProcessorError(
                "OpenAI SDK bulunamadı. Kurulum: pip install openai>=1.30.0"
            ) from e

        if not config.openai_api_key:
            raise AIProcessorError("OPENAI_API_KEY ortam değişkeni tanımlı değil.")

        from openai import OpenAI
        self._client = OpenAI(api_key=config.openai_api_key)
        self._config = config
        self._schema = AIAnalyzedReview.get_llm_schema()
        self._token_counter = TokenCounter(config.openai_model)

        # Structured Outputs için hazır JSON schema nesnesi
        self._json_schema_obj = {
            "name": "ai_analyzed_review",
            "strict": True,
            "schema": json.loads(self._schema),
        }
        logger.info(f"OpenAI provider başlatıldı: model={config.openai_model}")

    @property
    def model_name(self) -> str:
        return self._config.openai_model

    def _call_api(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        """
        OpenAI Chat Completion API çağrısı.
        Structured Outputs aktif: response_format=json_schema.
        Döndürür: (content_str, input_tokens, output_tokens)
        """
        for attempt in range(self._config.ai_max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.openai_model,
                    messages=messages,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": self._json_schema_obj,
                    },
                )
                content = response.choices[0].message.content or "{}"
                input_t = response.usage.prompt_tokens if response.usage else 0
                output_t = response.usage.completion_tokens if response.usage else 0
                logger.debug(
                    f"OpenAI yanıtı: "
                    f"tokens={input_t}+{output_t}, "
                    f"finish={response.choices[0].finish_reason}"
                )
                return content, input_t, output_t

            except self._RateLimitError as e:
                retry_after = int(
                    getattr(e, "retry_after", None) or 60
                )
                logger.warning(
                    f"OpenAI rate limit (deneme {attempt + 1}): "
                    f"{retry_after}s bekleniyor..."
                )
                time.sleep(retry_after)

            except self._APIConnectionError as e:
                logger.warning(
                    f"OpenAI bağlantı hatası (deneme {attempt + 1}): {e}"
                )
                _backoff_sleep(attempt)

            except self._APIStatusError as e:
                if e.status_code in {500, 502, 503, 529}:
                    logger.warning(
                        f"OpenAI sunucu hatası {e.status_code} "
                        f"(deneme {attempt + 1})"
                    )
                    _backoff_sleep(attempt)
                else:
                    # 4xx → retry'a değmez
                    raise AIAPIError("OpenAI", e.status_code, str(e)) from e

            except Exception as e:
                raise AIProcessorError(f"OpenAI beklenmeyen hata: {e}") from e

        raise AIAPIError(
            "OpenAI", None,
            f"{self._config.ai_max_retries} denemeden sonra başarısız."
        )

    def analyze_single(self, raw_text: str, source_id: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_analysis_prompt(raw_text, self._schema)},
        ]
        content, _, _ = self._call_api(messages)
        result = json.loads(content)
        result["source_id"] = source_id
        return result

    def analyze_batch(self, reviews: list[RawReviewData]) -> list[AIAnalyzedReview]:
        """
        RawReviewData listesini tek API çağrısında analiz eder.
        OpenAI Structured Outputs ile JSON şema constraint'lı yanıt alır,
        Pydantic ile doğrular ve AIAnalyzedReview listesi döndürür.
        """
        # Batch prompt için dict formatına çevir
        review_dicts = [
            {"source_id": r.source_id, "text": r.raw_text} for r in reviews
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_batch_prompt(review_dicts, self._schema)},
        ]
        content, _, _ = self._call_api(messages)
        parsed = json.loads(content)
        raw_list = (
            parsed.get("reviews", [parsed]) if isinstance(parsed, dict)
            else (parsed if isinstance(parsed, list) else [parsed])
        )
        results: list[AIAnalyzedReview] = []
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source_id", "unknown"))
            clean = {k: v for k, v in raw.items() if not k.startswith("_")}
            clean["source_id"] = source_id
            try:
                results.append(AIAnalyzedReview.model_validate(clean))
            except Exception as exc:
                logger.warning(
                    f"OpenAI batch validate hatası (source_id={source_id}): {exc}"
                )
        return results


# ---------------------------------------------------------------------------
# Anthropic Claude Provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseAIProvider):
    """
    Anthropic Claude modelleri için AI sağlayıcı.
    anthropic>=0.25.0 SDK kullanır.
    Claude'un markdown wrapping alışkanlığına karşı defensive JSON parse uygulanır.
    """

    def __init__(self, config: AIConfig) -> None:
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError as e:
            raise AIProcessorError(
                "Anthropic SDK bulunamadı. Kurulum: pip install anthropic>=0.25.0"
            ) from e
        if not config.anthropic_api_key:
            raise AIProcessorError("ANTHROPIC_API_KEY ortam değişkeni tanımlı değil.")
        self._client = self._anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._config = config
        self._schema = AIAnalyzedReview.get_llm_schema()
        self._token_counter = TokenCounter(config.anthropic_model)
        logger.info(f"Anthropic provider başlatıldı: model={config.anthropic_model}")

    @property
    def model_name(self) -> str:
        return self._config.anthropic_model

    def _extract_json(self, raw: str) -> str:
        """Claude'un ``` code block wrapping'ini temizler."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            raw = "\n".join(inner).strip()
        return raw

    def _call_api(self, user_content: str) -> tuple[str, int, int]:
        for attempt in range(self._config.ai_max_retries):
            try:
                message = self._client.messages.create(
                    model=self._config.anthropic_model,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                content = message.content[0].text
                return content, message.usage.input_tokens, message.usage.output_tokens
            except self._anthropic.RateLimitError:
                logger.warning(f"Anthropic rate limit (deneme {attempt+1}): 60s...")
                time.sleep(60)
            except self._anthropic.APIConnectionError as e:
                logger.warning(f"Anthropic bağlantı hatası (deneme {attempt+1}): {e}")
                _backoff_sleep(attempt)
            except self._anthropic.APIStatusError as e:
                if e.status_code in {500, 529}:
                    _backoff_sleep(attempt)
                else:
                    raise AIAPIError("Anthropic", e.status_code, str(e)) from e
            except Exception as e:
                raise AIProcessorError(f"Anthropic beklenmeyen hata: {e}") from e
        raise AIAPIError("Anthropic", None, f"{self._config.ai_max_retries} denemeden sonra başarısız.")

    def analyze_single(self, raw_text: str, source_id: str) -> dict[str, Any]:
        content, _, _ = self._call_api(build_analysis_prompt(raw_text, self._schema))
        result = json.loads(self._extract_json(content))
        result["source_id"] = source_id
        return result

    def analyze_batch(self, reviews: list[RawReviewData]) -> list[AIAnalyzedReview]:
        """
        RawReviewData listesini Anthropic Claude ile analiz eder.
        Markdown wrapping temizlenir, Pydantic ile doğrulanır.
        """
        review_dicts = [
            {"source_id": r.source_id, "text": r.raw_text} for r in reviews
        ]
        content, _, _ = self._call_api(build_batch_prompt(review_dicts, self._schema))
        parsed = json.loads(self._extract_json(content))
        raw_list = (
            parsed.get("reviews", [parsed]) if isinstance(parsed, dict)
            else (parsed if isinstance(parsed, list) else [parsed])
        )
        results: list[AIAnalyzedReview] = []
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source_id", "unknown"))
            clean = {k: v for k, v in raw.items() if not k.startswith("_")}
            clean["source_id"] = source_id
            try:
                results.append(AIAnalyzedReview.model_validate(clean))
            except Exception as exc:
                logger.warning(
                    f"Anthropic batch validate hatası (source_id={source_id}): {exc}"
                )
        return results


# ---------------------------------------------------------------------------
# Mock Provider - Akıllı Sentiment Simülasyonu
# ---------------------------------------------------------------------------

class MockAIProvider(BaseAIProvider):
    """
    Metin içeriğine duyarlı, deterministik mock AI sağlayıcı.

    Negatif sinyal kelime havuzu (EN+TR): broken, damaged, kötü, bozuk vb.
    Aynı source_id → her zaman aynı sonuç (test stabilitesi için seeded RNG).
    """

    _NEGATIVE_SIGNALS: frozenset[str] = frozenset({
        "broken", "damaged", "terrible", "awful", "horrible", "worst",
        "defective", "disappointed", "poor", "bad", "slow", "difficult",
        "wrong", "missing", "fail", "failed", "useless", "garbage",
        "refund", "return", "cracked", "fake", "dead", "stopped",
        "kötü", "bozuk", "berbat", "korkunç", "sorun", "çalışmıyor",
        "bozuldu", "kırık", "hatalı", "eksik", "rezalet", "pahalı",
    })
    _POSITIVE_SIGNALS: frozenset[str] = frozenset({
        "great", "excellent", "amazing", "love", "perfect", "fantastic",
        "outstanding", "best", "happy", "satisfied", "recommend", "good",
        "quality", "fast", "easy", "worth", "awesome", "superb", "reliable",
        "mükemmel", "harika", "süper", "memnun", "tavsiye",
    })
    _PRODUCT_NAMES: tuple[str, ...] = (
        "Wireless Bluetooth Headphones Pro X1",
        "Ergonomic Office Chair Model Z",
        "Stainless Steel Insulated Bottle 32oz",
        "Python Backend Development Masterclass",
        "Ultra HD 4K IPS Monitor 27-inch",
        "Compact Mechanical Keyboard TKL RGB",
        "Trail Running Shoes AeroFlex X3",
        "Premium Organic Green Tea Blend",
        "Smart Home Security Camera 1080p",
        "Portable Solar Power Bank 20000mAh",
    )
    _PROS_POOL: tuple[str, ...] = (
        "Excellent build quality and premium feel",
        "Fast shipping, well-packaged",
        "Battery life significantly exceeds expectations",
        "Setup was straightforward and quick",
        "Customer support responded within 2 hours",
        "Durable materials, shows no wear after 3 months",
        "Great value for the price point",
    )
    _CONS_POOL: tuple[str, ...] = (
        "User manual lacks sufficient detail",
        "Slightly smaller than product photos suggest",
        "Premium pricing may deter budget buyers",
        "Minor cosmetic blemish on arrival",
        "Packaging damaged during transit",
    )
    _TOPICS_POOL: tuple[str, ...] = (
        "build quality", "battery life", "value for money",
        "customer service", "shipping speed", "ease of use",
        "durability", "design aesthetics", "performance",
        "compatibility", "setup process",
    )

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._token_counter = TokenCounter("mock-v1.0-deterministic")
        logger.info("Mock AI provider başlatıldı (deterministik + içerik-duyarlı).")

    @property
    def model_name(self) -> str:
        return "mock-v1.0-deterministic"

    def _analyze_signals(
        self, text: str
    ) -> tuple[SentimentLabel, float, UrgencyLevel]:
        """Metin kelimelerini negatif/pozitif havuzlarla kesişim testi."""
        words = set(text.lower().split())
        neg = len(words & self._NEGATIVE_SIGNALS)
        pos = len(words & self._POSITIVE_SIGNALS)
        total = neg + pos or 1
        neg_ratio = neg / total

        if neg_ratio >= 0.7:
            return (
                SentimentLabel.NEGATIVE,
                round(-(0.4 + neg_ratio * 0.6), 3),
                UrgencyLevel.HIGH if neg_ratio >= 0.85 else UrgencyLevel.MEDIUM,
            )
        if neg_ratio >= 0.4:
            return SentimentLabel.MIXED, round(-neg_ratio * 0.4, 3), UrgencyLevel.MEDIUM
        if pos > 0 and neg == 0:
            score = round(min(0.3 + (pos / max(len(words), 1)) * 5, 0.95), 3)
            return SentimentLabel.POSITIVE, score, UrgencyLevel.LOW
        score = round(0.1 + (pos - neg) * 0.05, 3)
        label = SentimentLabel.POSITIVE if score > 0.15 else SentimentLabel.NEUTRAL
        return label, score, UrgencyLevel.LOW

    def analyze_single(self, raw_text: str, source_id: str) -> dict[str, Any]:
        seed = hash(source_id) % (2 ** 31)
        rng = random.Random(seed)
        sentiment, score, urgency = self._analyze_signals(raw_text)

        rating_ranges = {
            SentimentLabel.NEGATIVE: (1.0, 2.5),
            SentimentLabel.MIXED:    (2.5, 3.8),
            SentimentLabel.NEUTRAL:  (3.0, 4.0),
            SentimentLabel.POSITIVE: (3.5, 5.0),
        }
        lo, hi = rating_ranges[sentiment]
        rating = round(rng.uniform(lo, hi), 1)
        if sentiment == SentimentLabel.NEGATIVE and rating <= 2.0:
            urgency = UrgencyLevel.HIGH
        action = urgency in (UrgencyLevel.HIGH, UrgencyLevel.CRITICAL)

        product = rng.choice(self._PRODUCT_NAMES)
        return {
            "source_id": source_id,
            "product_name": product,
            "product_category": rng.choice(list(ProductCategory)).value,
            "product_id": f"SKU-{abs(seed) % 99999:05d}",
            "sentiment": sentiment.value,
            "sentiment_score": score,
            "confidence_score": round(rng.uniform(0.72, 0.97), 4),
            "review_date": None,
            "summary": (
                f"Customer expresses {sentiment.value} sentiment about {product}. "
                f"Rating: {rating}/5.0. "
                + ("Immediate follow-up required." if action else "No action needed.")
            ),
            "key_topics": rng.sample(list(self._TOPICS_POOL), k=rng.randint(2, 5)),
            "pros":  rng.sample(list(self._PROS_POOL), k=rng.randint(1, 3)),
            "cons":  rng.sample(list(self._CONS_POOL), k=rng.randint(0, 2)),
            "reviewer_rating": rating,
            "is_verified_purchase": rng.choice([True, True, False]),
            "reviewer_expertise": rng.choice(["novice", "intermediate", "expert"]),
            "urgency_level": urgency.value,
            "action_required": action,
            "action_notes": (
                "Flagged for customer support follow-up within 24 hours."
                if action else None
            ),
            "mentioned_competitors": (
                rng.sample(["CompetitorAlpha", "BrandXPro", "AltProduct"], k=1)
                if rng.random() < 0.12 else []
            ),
            "data_quality": rng.choice([
                DataQuality.HIGH.value, DataQuality.HIGH.value, DataQuality.MEDIUM.value
            ]),
            "language_detected": "en",
            "is_spam_or_fake": rng.random() < 0.04,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_batch(
        self, reviews: list["RawReviewData"]
    ) -> list["AIAnalyzedReview"]:
        """
        RawReviewData listesini alır, her biri için analyze_single çağırır
        ve doğrulanmış AIAnalyzedReview listesi döndürür.
        """
        results: list[AIAnalyzedReview] = []
        for record in reviews:
            raw = self.analyze_single(record.raw_text, record.source_id)
            # Internal meta alanları (_*) filtrele, source_id ekle
            clean = {k: v for k, v in raw.items() if not k.startswith("_")}
            clean["source_id"] = record.source_id
            try:
                results.append(AIAnalyzedReview.model_validate(clean))
            except Exception as exc:
                logger.warning(
                    f"MockAIProvider validate hatası (source_id={record.source_id}): {exc}"
                )
        return results


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

def create_ai_provider(config: AIConfig) -> BaseAIProvider:
    """Factory: AI_PROVIDER değerine göre uygun provider örneği döner."""
    registry: dict[str, type[BaseAIProvider]] = {
        "openai":    OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock":      MockAIProvider,
    }
    cls = registry.get(config.provider)
    if cls is None:
        raise AIProcessorError(
            f"Desteklenmeyen provider: '{config.provider}'. "
            f"Geçerliler: {', '.join(registry)}"
        )
    logger.info(f"AI Provider seçildi: {config.provider.upper()}")
    return cls(config)


# ---------------------------------------------------------------------------
# Token-Aware Batch Splitter
# ---------------------------------------------------------------------------

class BatchSplitter:
    """
    Kayıt listesini token bütçesine göre sub-batch'lere böler.
    Context window taşmasını önler; maliyet optimize edilir.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens_per_batch: int = 6000,
        schema_overhead: int = 800,
    ) -> None:
        self._counter = token_counter
        self._max = max_tokens_per_batch
        self._overhead = schema_overhead

    def split(self, records: list[RawReviewData]) -> list[list[RawReviewData]]:
        batches: list[list[RawReviewData]] = []
        current: list[RawReviewData] = []
        current_tokens = self._overhead

        for record in records:
            record_tokens = self._counter.count(record.raw_text) + 50
            if current and current_tokens + record_tokens > self._max:
                batches.append(current)
                current = [record]
                current_tokens = self._overhead + record_tokens
            else:
                current.append(record)
                current_tokens += record_tokens

        if current:
            batches.append(current)

        logger.debug(
            f"BatchSplitter: {len(records)} kayıt → "
            f"{len(batches)} sub-batch (max_tokens={self._max})"
        )
        return batches


# ---------------------------------------------------------------------------
# Ana AI İşleme Orkestratörü
# ---------------------------------------------------------------------------

class AIProcessor:
    """
    Ham RawReviewData listesini AIBatchResult'a dönüştüren orkestratör.
    Token-aware splitting + retry + Pydantic validation + cost tracking.
    """

    def __init__(self, config: AIConfig) -> None:
        self._config = config
        self._provider = create_ai_provider(config)
        self._schema = AIAnalyzedReview.get_llm_schema()

        model_for_counter = {
            "openai": config.openai_model,
            "anthropic": config.anthropic_model,
        }.get(config.provider, "mock-v1.0-deterministic")

        self._token_counter = TokenCounter(model_for_counter)
        self._batch_splitter = BatchSplitter(
            self._token_counter,
            max_tokens_per_batch=getattr(config, "max_tokens_per_batch", 6000),
        )
        logger.info(
            f"AIProcessor hazır: provider={config.provider}, "
            f"model={self._provider.model_name}, "
            f"batch_size={config.batch_size}"
        )

    def _parse_and_validate(
        self, raw_dict: dict[str, Any], source_id: str
    ) -> Optional[AIAnalyzedReview]:
        """Internal meta-alanları (_*) temizler ve Pydantic ile doğrular."""
        clean = {k: v for k, v in raw_dict.items() if not k.startswith("_")}
        clean["source_id"] = source_id
        try:
            return AIAnalyzedReview.model_validate(clean)
        except Exception as e:
            logger.warning(
                f"Pydantic doğrulama başarısız (source_id={source_id}): {e}"
            )
            return None

    def _process_single_with_retry(
        self, record: RawReviewData
    ) -> Optional[AIAnalyzedReview]:
        for attempt in range(self._config.ai_max_retries):
            try:
                raw = self._provider.analyze_single(record.raw_text, record.source_id)
                return self._parse_and_validate(raw, record.source_id)
            except AIRateLimitError as e:
                time.sleep(e.retry_after)
            except (AIAPIError, AIProcessorError) as e:
                logger.warning(f"AI hatası (deneme {attempt+1}): {e}")
                _backoff_sleep(attempt)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse hatası (deneme {attempt+1}): {e}")
                _backoff_sleep(attempt)
        logger.error(f"✗ Tüm denemeler tükendi: source_id={record.source_id}")
        return None

    def _process_batch_records(
        self, batch: list[RawReviewData]
    ) -> tuple[list[AIAnalyzedReview], list[str]]:
        """
        Bir sub-batch'i provider'a gönderir.
        MockAIProvider dahil tüm provider'lar artık list[RawReviewData] alıp
        list[AIAnalyzedReview] döndürür.
        """
        successful: list[AIAnalyzedReview] = []
        failed_ids: list[str] = []

        try:
            # analyze_batch() → list[AIAnalyzedReview] (zaten validate edilmiş)
            results = self._provider.analyze_batch(batch)
            for item in results:
                if isinstance(item, AIAnalyzedReview):
                    successful.append(item)
                elif isinstance(item, dict):
                    # Geriye dönük uyumluluk: dict döndüren provider'lar için
                    source_id = item.get("source_id", "unknown")
                    v = self._parse_and_validate(item, source_id)
                    if v:
                        successful.append(v)
                    else:
                        failed_ids.append(source_id)
            # Başarısız olan kayıtları single retry ile tamamla
            succeeded_ids = {r.source_id for r in successful}
            for record in batch:
                if record.source_id not in succeeded_ids:
                    single = self._process_single_with_retry(record)
                    if single:
                        successful.append(single)
                    else:
                        failed_ids.append(record.source_id)
        except (AIAPIError, AIProcessorError, json.JSONDecodeError) as e:
            logger.warning(f"Batch başarısız ({e}), single mod'a düşülüyor...")
            for record in batch:
                single = self._process_single_with_retry(record)
                if single:
                    successful.append(single)
                else:
                    failed_ids.append(record.source_id)

        return successful, failed_ids

    def process_all(self, raw_records: list[RawReviewData]) -> AIBatchResult:
        """
        Token-aware batch splitting ile tüm kayıtları senkron olarak işler.
        Sub-batch'ler sırayla çalıştırılır; hafıza kullanımı öngörülebilir.
        """
        batch_id = str(uuid.uuid4())[:8]
        total = len(raw_records)

        logger.info(
            f"━━━ AI İşleme Başlıyor ━━━ "
            f"batch_id={batch_id} | toplam={total} | "
            f"provider={self._config.provider.upper()}"
        )
        start = time.time()

        sub_batches = self._batch_splitter.split(raw_records)
        logger.info(
            f"Token-aware splitting: {total} kayıt → {len(sub_batches)} sub-batch"
        )

        all_successful: list[AIAnalyzedReview] = []
        all_failed: list[str] = []
        total_tokens = 0

        for idx, sub_batch in enumerate(sub_batches, start=1):
            logger.info(
                f"Sub-batch {idx}/{len(sub_batches)} işleniyor "
                f"({len(sub_batch)} kayıt)..."
            )
            batch_tokens = sum(
                self._token_counter.count(r.raw_text) for r in sub_batch
            )
            total_tokens += batch_tokens
            ok, failed = self._process_batch_records(sub_batch)
            all_successful.extend(ok)
            all_failed.extend(failed)
            logger.info(
                f"Sub-batch {idx}: ✓{len(ok)} / ✗{len(failed)} | ~{batch_tokens} token"
            )

        elapsed = round(time.time() - start, 2)
        cost = self._token_counter.estimate_cost_usd(
            input_tokens=total_tokens,
            output_tokens=len(all_successful) * 200,
        )

        result = AIBatchResult(
            batch_id=batch_id,
            total_input=total,
            successful=all_successful,
            failed_ids=all_failed,
            processing_time_seconds=elapsed,
            model_used=self._provider.model_name,
            total_tokens_used=total_tokens,
            estimated_cost_usd=cost,
        )

        logger.info(
            f"━━━ AI Tamamlandı ━━━ "
            f"başarı={result.success_rate:.1%} | "
            f"✓{len(all_successful)} / ✗{len(all_failed)} | "
            f"~{total_tokens} token | ~${cost:.4f} | {elapsed}s"
        )
        return result

    # -----------------------------------------------------------------------
    # Asenkron İşleme (Opsiyonel - Büyük veri setleri için)
    # -----------------------------------------------------------------------

    async def _process_batch_records_async(
        self,
        batch: list[RawReviewData],
        batch_idx: int,
    ) -> tuple[list[AIAnalyzedReview], list[str], int]:
        """
        Tek bir sub-batch'i asyncio event loop'unda işler.
        CPU-bound işlemi engellemek için run_in_executor kullanılır.
        Döndürür: (successful, failed_ids, token_count)
        """
        loop = asyncio.get_event_loop()

        # _process_batch_records senkron → thread pool'da çalıştır
        ok, failed = await loop.run_in_executor(
            None, self._process_batch_records, batch
        )
        batch_tokens = sum(self._token_counter.count(r.raw_text) for r in batch)

        logger.info(
            f"Async sub-batch {batch_idx}: "
            f"✓{len(ok)} / ✗{len(failed)} | ~{batch_tokens} token"
        )
        return ok, failed, batch_tokens

    async def process_all_async(
        self,
        raw_records: list[RawReviewData],
        max_concurrent: int = 3,
    ) -> AIBatchResult:
        """
        Token-aware batch splitting + asyncio.gather ile paralel sub-batch işleme.

        Avantajları:
          - Ağ I/O bekleme sürelerinde diğer sub-batch'ler devam eder
          - max_concurrent ile rate limit koruması sağlanır (Semaphore)
          - Mock provider ile anında döner; gerçek API'lerde ~N×hız artışı

        Parametreler:
          raw_records    : İşlenecek ham kayıtlar
          max_concurrent : Eşzamanlı çalışacak maksimum sub-batch sayısı
                           (default=3; OpenAI RPM limitine göre ayarlanmalı)

        Döndürür:
          AIBatchResult — process_all() ile aynı tip
        """
        batch_id = str(uuid.uuid4())[:8]
        total = len(raw_records)

        logger.info(
            f"━━━ Async AI İşleme Başlıyor ━━━ "
            f"batch_id={batch_id} | toplam={total} | "
            f"provider={self._config.provider.upper()} | "
            f"max_concurrent={max_concurrent}"
        )
        start = time.time()

        sub_batches = self._batch_splitter.split(raw_records)
        logger.info(
            f"Token-aware splitting: {total} kayıt → {len(sub_batches)} sub-batch "
            f"(paralel, maks {max_concurrent} eşzamanlı)"
        )

        # Semaphore: eşzamanlı çalışan görev sayısını sınırlar (rate limit koruması)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded(batch: list[RawReviewData], idx: int):
            async with semaphore:
                return await self._process_batch_records_async(batch, idx)

        # Tüm sub-batch'leri paralel başlat
        tasks = [
            _bounded(sub_batch, idx)
            for idx, sub_batch in enumerate(sub_batches, start=1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Sonuçları birleştir
        all_successful: list[AIAnalyzedReview] = []
        all_failed: list[str] = []
        total_tokens = 0

        for ok, failed, tokens in results:
            all_successful.extend(ok)
            all_failed.extend(failed)
            total_tokens += tokens

        elapsed = round(time.time() - start, 2)
        cost = self._token_counter.estimate_cost_usd(
            input_tokens=total_tokens,
            output_tokens=len(all_successful) * 200,
        )

        result = AIBatchResult(
            batch_id=batch_id,
            total_input=total,
            successful=all_successful,
            failed_ids=all_failed,
            processing_time_seconds=elapsed,
            model_used=self._provider.model_name,
            total_tokens_used=total_tokens,
            estimated_cost_usd=cost,
        )

        logger.info(
            f"━━━ Async AI Tamamlandı ━━━ "
            f"başarı={result.success_rate:.1%} | "
            f"✓{len(all_successful)} / ✗{len(all_failed)} | "
            f"~{total_tokens} token | ~${cost:.4f} | {elapsed}s"
        )
        return result
