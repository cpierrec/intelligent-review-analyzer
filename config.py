"""
config.py - Merkezi Yapılandırma Modülü
=======================================
Amaç: Tüm sistem genelinde kullanılan yapılandırma parametrelerini,
ortam değişkenlerini ve sabit değerleri tek bir yerden yönetmek.
Bu yaklaşım (12-Factor App prensibi), kodu ortamdan bağımsız kılar
ve güvenlik açısından API anahtarlarının kaynak koduna gömülmesini önler.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# .env dosyasını yükle (varsa); production'da sistem ortam değişkenleri kullanılır
load_dotenv()


# ---------------------------------------------------------------------------
# Loglama Altyapısı - Tüm modüller bu yapılandırmayı miras alır
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Merkezi loglama yapılandırması.
    Hem konsola (StreamHandler) hem de dosyaya (FileHandler) yazar.
    Üretim ortamında dosya rotasyonu (RotatingFileHandler) önerilir.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Log formatı: zaman damgası + modül adı + seviye + mesaj
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Kök logger'ı yapılandır
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Mevcut handler'ları temizle (tekrar çağrılma durumuna karşı)
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Konsol çıktısı
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Dosya çıktısı - hata ayıklama ve audit trail için kritik
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/engine.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


# ---------------------------------------------------------------------------
# Ağ Katmanı Yapılandırması
# ---------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    """
    Dirençli ağ katmanı için retry stratejisi ve timeout parametreleri.
    Exponential backoff ile geçici ağ hatalarını (5xx, ConnectionError) yönetir.
    """
    # Toplam yeniden deneme sayısı (RFC 7230 uyumlu güvenli üst sınır)
    max_retries: int = int(os.getenv("NET_MAX_RETRIES", "5"))

    # İlk bekleme süresi (saniye) - her denemede 2x artar (backoff_factor)
    backoff_factor: float = float(os.getenv("NET_BACKOFF_FACTOR", "0.5"))

    # Bağlantı ve okuma timeout'ları (saniye) - tuple: (connect_timeout, read_timeout)
    request_timeout: tuple[int, int] = field(default_factory=lambda: (10, 30))

    # Retry tetiklenecek HTTP durum kodları
    retry_status_codes: frozenset = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )

    # Eşzamanlı istek sınırı (rate limiting için)
    max_concurrent_requests: int = int(os.getenv("NET_MAX_CONCURRENT", "5"))

    # Kibarca davran: İstekler arası minimum bekleme (saniye)
    request_delay: float = float(os.getenv("NET_REQUEST_DELAY", "1.0"))

    # User-Agent başlığı - gerçekçi bir tarayıcı imzası
    user_agent: str = (
        "Mozilla/5.0 (compatible; AIDataEngine/1.0; "
        "+https://github.com/portfolio/ai-data-engine)"
    )


# ---------------------------------------------------------------------------
# AI Servis Yapılandırması
# ---------------------------------------------------------------------------
@dataclass
class AIConfig:
    """
    LLM API entegrasyon parametreleri.
    Hem OpenAI hem de Anthropic Claude desteklenir;
    AI_PROVIDER ortam değişkeni ile seçim yapılır.
    """
    # Kullanılacak AI sağlayıcısı: "openai" veya "anthropic"
    provider: str = os.getenv("AI_PROVIDER", "mock").lower()

    # API anahtarları - ASLA kaynak koduna gömme, .env veya secret manager kullan
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Model seçimleri - maliyet/performans dengesi için yapılandırılabilir
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    # Token limitleri ve sıcaklık parametresi
    max_tokens: int = int(os.getenv("AI_MAX_TOKENS", "1500"))
    temperature: float = float(os.getenv("AI_TEMPERATURE", "0.1"))  # Düşük = tutarlı çıktı

    # Batch processing: tek seferde işlenecek yorum sayısı
    batch_size: int = int(os.getenv("AI_BATCH_SIZE", "5"))

    # AI çağrısı başarısız olursa yeniden deneme
    ai_max_retries: int = int(os.getenv("AI_MAX_RETRIES", "3"))


# ---------------------------------------------------------------------------
# Veri Kaynağı Yapılandırması
# ---------------------------------------------------------------------------
@dataclass
class DataSourceConfig:
    """
    Veri kazıma hedeflerini ve mock veri üretim parametrelerini tanımlar.
    Gerçek bir projede bu değerler bir veritabanından veya config dosyasından gelir.
    """
    # Mock API endpoint'i (httpbin.org - public test servisi)
    mock_api_base_url: str = "https://jsonplaceholder.typicode.com"

    # Hedef kaynak türü: "mock_api", "html_scrape", "rss_feed"
    source_type: str = os.getenv("DATA_SOURCE_TYPE", "mock_api")

    # Maksimum çekilecek kayıt sayısı
    max_records: int = int(os.getenv("MAX_RECORDS", "20"))

    # Ham veri önbellek dizini
    raw_data_cache_dir: str = "data/raw"

    # İşlenmiş veri çıktı dizini
    processed_data_dir: str = "data/processed"


# ---------------------------------------------------------------------------
# Raporlama Yapılandırması
# ---------------------------------------------------------------------------
@dataclass
class ReportConfig:
    """
    Çıktı formatları ve raporlama dizinleri için yapılandırma.
    """
    output_dir: str = os.getenv("REPORT_OUTPUT_DIR", "reports")

    # Çıktı formatları: json, csv, xlsx
    export_formats: list = field(
        default_factory=lambda: ["json", "csv", "xlsx"]
    )

    # Excel raporu için sayfa adı
    excel_sheet_name: str = "AI_Analysis_Report"

    # JSON çıktısı için indentation (okunabilirlik)
    json_indent: int = 2

    # Tarih-saat damgası rapor adına eklensin mi?
    timestamp_filenames: bool = True


# ---------------------------------------------------------------------------
# Ana Uygulama Yapılandırması - Singleton benzeri merkezi erişim noktası
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    """
    Tüm alt yapılandırmaları birleştiren ana yapılandırma nesnesi.
    Dependency injection ile modüllere aktarılır.
    """
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    app_version: str = "1.0.0"
    app_name: str = "AI-Powered Data Integration Engine"

    def validate(self) -> None:
        """
        Başlangıçta kritik yapılandırmaları doğrular.
        Hatalı yapılandırma ile çalışmayı önler (fail-fast prensibi).
        """
        logger = logging.getLogger(__name__)

        if self.ai.provider == "openai" and not self.ai.openai_api_key:
            raise EnvironmentError(
                "AI_PROVIDER=openai seçildi fakat OPENAI_API_KEY tanımlı değil. "
                "Lütfen .env dosyanıza ekleyin veya 'mock' provider kullanın."
            )

        if self.ai.provider == "anthropic" and not self.ai.anthropic_api_key:
            raise EnvironmentError(
                "AI_PROVIDER=anthropic seçildi fakat ANTHROPIC_API_KEY tanımlı değil. "
                "Lütfen .env dosyanıza ekleyin veya 'mock' provider kullanın."
            )

        if self.ai.provider == "mock":
            logger.warning(
                "AI sağlayıcı 'mock' modunda çalışıyor. "
                "Gerçek analiz için OPENAI_API_KEY veya ANTHROPIC_API_KEY tanımlayın."
            )

        # Gerekli dizinleri oluştur
        for directory in [
            self.data_source.raw_data_cache_dir,
            self.data_source.processed_data_dir,
            self.report.output_dir,
            "logs",
        ]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Dizin hazır: {directory}")

        logger.info(
            f"{self.app_name} v{self.app_version} yapılandırması doğrulandı. "
            f"AI Provider: {self.ai.provider.upper()}"
        )


# Modül düzeyinde varsayılan yapılandırma örneği (convenience import için)
default_config = AppConfig()
