"""
models.py - Pydantic Veri Modelleri ve JSON Şema Katmanı
=========================================================
Amaç: Sistemdeki tüm veri yapılarını Pydantic modelleri ile tanımlamak.
Bu yaklaşım üç kritik avantaj sağlar:
  1. Runtime type validation  — AI'dan gelen JSON çıktısı otomatik doğrulanır;
     production'da tip hataları sıfıra iner.
  2. Otomatik JSON şema üretimi — LLM prompt'larına şema enjekte edilir;
     prompt mühendisliği standartlaşır, schema drift önlenir.
  3. Serialization/deserialization — dict ↔ model dönüşümü hata payı sıfır;
     pandas, JSON, Excel katmanları aynı doğrulanmış veriyi tüketir.

Domain: E-Ticaret Ürün Yorum Analizi (Product Review Intelligence)
Use-case: Binlerce müşteri yorumundan güvenilir, aksiyon alınabilir iş
          zekası üretmek; customer-support önceliklendirme otomasyonu.

Tasarım Kararları:
  - str Enum mixin → JSON serialize edildiğinde string değer (int yerine)
  - Field(examples=[...]) → OpenAPI / JSON Schema uyumlu örnek değerler
  - @field_validator(mode="before") → LLM'den gelen ham dict hataları yakalar
  - @model_validator(mode="after") → Cross-field iş kuralları (urgency otomasyonu)
  - ClassVar[str] ile schema cache → get_llm_schema() çağrısı CPU maliyetsiz
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enum Tanımları - Kategorik alanlar için tip güvenliği
# ---------------------------------------------------------------------------

class SentimentLabel(str, Enum):
    """
    Duygu analizi sonucu.
    str mixin'i JSON serialization'ı doğrudan string olarak yapar.
    """
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class ProductCategory(str, Enum):
    """Ürün kategorileri - gerçek projede veritabanından çekilir."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HOME_GARDEN = "home_garden"
    SPORTS = "sports"
    BOOKS = "books"
    BEAUTY = "beauty"
    UNKNOWN = "unknown"


class UrgencyLevel(str, Enum):
    """
    Müşteri şikayetinin aciliyet seviyesi.
    Customer support önceliklendirme için kullanılır.
    """
    CRITICAL = "critical"    # Ürün iade/hukuki risk içeriyor
    HIGH = "high"            # 24 saat içinde yanıt gerektirir
    MEDIUM = "medium"        # 3 iş günü içinde yanıt
    LOW = "low"              # Genel geri bildirim


class DataQuality(str, Enum):
    """Ham verinin kalite skoru - analitikte filtreleme için kullanılır."""
    HIGH = "high"        # Tam ve güvenilir
    MEDIUM = "medium"    # Eksik alanlar var ama kullanılabilir
    LOW = "low"          # Çok eksik, analitikten hariç tutulabilir


# ---------------------------------------------------------------------------
# Ham Veri Modeli - Scraper katmanının çıktısı
# ---------------------------------------------------------------------------

class RawReviewData(BaseModel):
    """
    Web scraper'ın ürettiği ham, yapılandırılmamış veri.
    AI işlemeden geçmeden önce bu forma normalize edilir.

    Invariant: source_id boş olamaz; raw_text en az 10 karakter içermelidir.
    Bu kısıtlar AI katmanına anlamsız / boş veri gitmesini engeller.
    """

    source_id: str = Field(
        ...,
        min_length=1,
        description="Kaynak sistemdeki benzersiz ID. Boş string kabul edilmez.",
        examples=["post_42", "comment_1337", "html_7"],
    )
    source_url: Optional[str] = Field(
        None,
        description="Verinin çekildiği tam URL (audit trail için).",
        examples=["https://jsonplaceholder.typicode.com/posts/1/comments"],
    )
    raw_text: str = Field(
        ...,
        min_length=10,
        description="Ham yorum metni. AI prompt'una doğrudan iletilir.",
        examples=["Title: Great product\nReview: Battery life exceeds expectations."],
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Veri çekme zaman damgası (UTC, timezone-aware).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Kaynak'a özgü ek meta veriler (star_rating, price vb.).",
        examples=[{"star_rating": 4, "price": "$29.99", "availability": "In Stock"}],
    )

    @field_validator("source_id", mode="before")
    @classmethod
    def validate_source_id_not_empty(cls, v: Any) -> str:
        """
        source_id boş string, None veya sadece boşluk olmamalı.
        Pipeline boyunca join/dedup operasyonları bu alana dayanır.
        """
        stripped = str(v).strip() if v is not None else ""
        if not stripped:
            raise ValueError(
                "source_id boş olamaz. Kaynak sistemdeki benzersiz ID zorunludur."
            )
        return stripped

    @field_validator("raw_text", mode="before")
    @classmethod
    def validate_and_clean_raw_text(cls, v: Any) -> str:
        """
        Ham metni strip eder ve minimum uzunluk kontrolü yapar.
        10 karakterden kısa metinler AI analizine değmez; erken elenir.
        """
        if not isinstance(v, str):
            v = str(v)
        cleaned = v.strip()
        if len(cleaned) < 10:
            raise ValueError(
                f"raw_text çok kısa ({len(cleaned)} karakter). "
                "Anlamlı analiz için en az 10 karakter gereklidir."
            )
        return cleaned


# ---------------------------------------------------------------------------
# AI Çıktı Modeli - LLM'in üretmesi beklenen yapılandırılmış veri
# ---------------------------------------------------------------------------

class AIAnalyzedReview(BaseModel):
    """
    AI katmanının ham metinden çıkardığı yapılandırılmış iş zekası.

    Bu model aynı zamanda LLM prompt'una enjekte edilen JSON şemasını üretir.
    Pydantic'in model_json_schema() metodu ile otomatik şema alınır,
    bu da prompt mühendisliğini standartlaştırır ve tutarsız AI çıktılarını önler.

    Validation Hierarchy:
      1. Field-level (ge/le/min_length) → Pydantic otomatik uygular
      2. @field_validator            → LLM çıktısının edge case'leri
      3. @model_validator(after)     → Cross-field iş kuralları
    """

    # Schema cache: get_llm_schema() ilk çağrıda hesaplanır, sonrası cached.
    _schema_cache: ClassVar[Optional[str]] = None

    # --- Kaynak Bilgisi ---
    source_id: str = Field(
        ...,
        min_length=1,
        description="Ham veriden gelen orijinal kayıt kimliği. Pipeline boyunca değişmez.",
        examples=["comment_42", "post_7_comment_3", "html_12"],
    )

    # --- Ürün Bilgisi ---
    product_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Yorumda bahsedilen ürün veya hizmet adı.",
        examples=["Wireless Bluetooth Headphones Pro", "Ergonomic Office Chair Deluxe"],
    )
    product_category: ProductCategory = Field(
        default=ProductCategory.UNKNOWN,
        description="Ürün kategorisi. Bilinmiyorsa 'unknown' atanır.",
        examples=["electronics", "clothing", "books"],
    )
    product_id: Optional[str] = Field(
        None,
        description="Metinden çıkarılabilen SKU/ASIN/ürün kodu. Bulunamazsa null.",
        examples=["SKU-00042", "B08N5WRWNW", None],
    )

    # --- Duygu Analizi ---
    sentiment: SentimentLabel = Field(
        ...,
        description="Yorumun genel duygu etiketi.",
        examples=["positive", "negative", "neutral", "mixed"],
    )
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Sürekli duygu skoru. "
            "-1.0 = tamamen negatif, 0.0 = nötr, +1.0 = tamamen pozitif."
        ),
        examples=[0.85, -0.62, 0.0, 0.23],
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "AI analizinin güven skoru (0.0-1.0). "
            "0.5 altı düşük kalite olarak işaretlenir."
        ),
        examples=[0.92, 0.75, 0.51],
    )

    # --- Zaman Bilgisi (Time-Series Analizi için kritik) ---
    review_date: Optional[datetime] = Field(
        None,
        description=(
            "Yorumun orijinal yazılma tarihi (UTC). "
            "Time-series analizi ve trend tespiti için kullanılır."
        ),
        examples=["2024-03-15T10:30:00Z", None],
    )

    # --- İçerik Analizi ---
    summary: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Yorumun 1-2 cümlelik, iş odaklı özeti. Yönetici raporlarında kullanılır.",
        examples=[
            "Customer is highly satisfied with battery life but notes setup difficulty.",
            "Product arrived damaged; customer requests immediate replacement.",
        ],
    )
    key_topics: list[str] = Field(
        default_factory=list,
        description="Yorumda geçen ana konular/özellikler (maks 10).",
        examples=[["battery life", "build quality", "value for money"]],
    )
    pros: list[str] = Field(
        default_factory=list,
        description="Müşterinin açıkça olumlu bulduğu noktalar.",
        examples=[["Fast shipping", "Excellent build quality", "Easy setup"]],
    )
    cons: list[str] = Field(
        default_factory=list,
        description="Müşterinin açıkça olumsuz bulduğu noktalar.",
        examples=[["Instructions unclear", "Price too high"]],
    )

    # --- Müşteri Bilgisi ---
    reviewer_rating: Optional[float] = Field(
        None,
        description=(
            "Yorumdan çıkarılan yıldız puanı (1.0-5.0 arası). "
            "Bulunamazsa null; 0 veya 6+ geçersiz sayılır."
        ),
        examples=[5.0, 3.5, 1.0, None],
    )
    is_verified_purchase: Optional[bool] = Field(
        None,
        description="Doğrulanmış satın alma göstergesi. Bilinmiyorsa null.",
        examples=[True, False, None],
    )
    reviewer_expertise: Optional[str] = Field(
        None,
        description="Yorumcunun konuya hakimiyeti: 'novice', 'intermediate' veya 'expert'.",
        examples=["expert", "intermediate", "novice", None],
    )

    # --- İş Zekası ---
    urgency_level: UrgencyLevel = Field(
        default=UrgencyLevel.LOW,
        description=(
            "Müşteri hizmetleri aksiyon önceliği. "
            "Model validator tarafından sentiment+rating kombinasyonuna göre otomatik yükseltilir."
        ),
        examples=["critical", "high", "medium", "low"],
    )
    action_required: bool = Field(
        default=False,
        description="Customer support ekibinin bu yoruma aksiyon alması gerekiyor mu?",
        examples=[True, False],
    )
    action_notes: Optional[str] = Field(
        None,
        max_length=300,
        description="Önerilen aksiyon notları (customer support için).",
        examples=["Contact customer within 24h regarding damaged product.", None],
    )
    mentioned_competitors: list[str] = Field(
        default_factory=list,
        description="Yorumda geçen rakip marka veya ürün adları.",
        examples=[["CompetitorX Pro", "BrandY Deluxe"], []],
    )

    # --- Kalite Kontrolü ---
    data_quality: DataQuality = Field(
        default=DataQuality.MEDIUM,
        description=(
            "Bu analizin güvenilirlik kalitesi. "
            "'low' kaliteli kayıtlar aggregate analizden hariç tutulabilir."
        ),
        examples=["high", "medium", "low"],
    )
    language_detected: str = Field(
        default="en",
        min_length=2,
        max_length=5,
        description="Yorumun ISO 639-1 dil kodu.",
        examples=["en", "tr", "de", "fr"],
    )
    is_spam_or_fake: bool = Field(
        default=False,
        description="AI tarafından spam veya sahte yorum olarak tespit edildi mi?",
        examples=[False, True],
    )

    # --- Zaman Damgası ---
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="AI analiz zaman damgası (UTC, timezone-aware).",
    )

    # -----------------------------------------------------------------------
    # Field Validators
    # -----------------------------------------------------------------------

    @field_validator("source_id", mode="before")
    @classmethod
    def validate_source_id(cls, v: Any) -> str:
        """source_id boş veya sadece whitespace olamaz."""
        stripped = str(v).strip() if v is not None else ""
        if not stripped:
            raise ValueError("source_id boş olamaz.")
        return stripped

    @field_validator("reviewer_rating", mode="before")
    @classmethod
    def validate_rating_range(cls, v: Any) -> Optional[float]:
        """
        Rating kesinlikle 1.0-5.0 arasında olmalı.
        0 yıldız veya 6 yıldız gibi geçersiz değerler None'a dönüştürülür;
        exception fırlatmak yerine graceful null tercih edilir (LLM toleransı).
        """
        if v is None:
            return None
        try:
            rating = float(v)
        except (TypeError, ValueError):
            return None  # Dönüştürülemeyen değer → null
        if not (1.0 <= rating <= 5.0):
            # Geçersiz aralık: production'da hata yerine null döndür
            return None
        return round(rating, 1)

    @field_validator("confidence_score", mode="before")
    @classmethod
    def validate_confidence_range(cls, v: Any) -> float:
        """
        confidence_score 0.0-1.0 arasında zorunlu.
        LLM bazen yüzde formatında (0-100) döndürebilir; normalize edilir.
        """
        if v is None:
            raise ValueError("confidence_score null olamaz.")
        try:
            score = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"confidence_score sayısal olmalı, '{v}' geçersiz.") from exc
        # LLM 0-100 skala kullandıysa normalize et
        if score > 1.0 and score <= 100.0:
            score = score / 100.0
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"confidence_score 0.0-1.0 arasında olmalı, alınan: {score}"
            )
        return round(score, 4)

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def validate_sentiment_score_range(cls, v: Any) -> float:
        """sentiment_score -1.0 ile +1.0 arasına clip eder."""
        try:
            score = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sentiment_score sayısal olmalı, '{v}' geçersiz.") from exc
        # Soft clip: aralık dışındaki değerleri sınırla, exception fırlatma
        return round(max(-1.0, min(1.0, score)), 4)

    @field_validator("reviewer_expertise", mode="before")
    @classmethod
    def validate_expertise_values(cls, v: Any) -> Optional[str]:
        """Sadece izin verilen 3 değeri kabul et; bilinmeyeni None yap."""
        if v is None:
            return None
        allowed = {"novice", "intermediate", "expert"}
        normalized = str(v).strip().lower()
        return normalized if normalized in allowed else None

    @field_validator("key_topics", "pros", "cons", "mentioned_competitors", mode="before")
    @classmethod
    def ensure_string_list(cls, v: Any) -> list[str]:
        """
        LLM bazen virgülle ayrılmış string döndürür; listeye çevir.
        None veya boş değerleri güvenle ele al.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("language_detected", mode="before")
    @classmethod
    def normalize_language_code(cls, v: Any) -> str:
        """Dil kodunu lowercase 2-karakter ISO 639-1 formatına normalize et."""
        if not v:
            return "en"
        return str(v).strip().lower()[:5]  # max 5 char (zh-CN gibi)

    # -----------------------------------------------------------------------
    # Cross-Field Business Rules
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def apply_business_rules(self) -> "AIAnalyzedReview":
        """
        Cross-field iş kuralları — field validator'lardan SONRA çalışır:

        Kural 1: Negatif sentiment + rating ≤ 2.0 → urgency=HIGH, action_required=True
        Kural 2: CRITICAL urgency → action_required her zaman True
        Kural 3: is_spam_or_fake=True → data_quality=LOW otomatik atanır
        """
        # Kural 1: Düşük puanlı negatif yorum → HIGH urgency
        if (
            self.sentiment == SentimentLabel.NEGATIVE
            and self.reviewer_rating is not None
            and self.reviewer_rating <= 2.0
            and self.urgency_level == UrgencyLevel.LOW
        ):
            self.urgency_level = UrgencyLevel.HIGH
            self.action_required = True

        # Kural 2: CRITICAL → action zorunlu
        if self.urgency_level == UrgencyLevel.CRITICAL:
            self.action_required = True

        # Kural 3: Spam → otomatik düşük kalite
        if self.is_spam_or_fake:
            self.data_quality = DataQuality.LOW

        return self

    # -----------------------------------------------------------------------
    # Schema Utilities
    # -----------------------------------------------------------------------

    @classmethod
    def get_llm_schema(cls) -> str:
        """
        LLM prompt'una enjekte edilecek JSON şemasını üretir.
        ClassVar cache ile tekrarlayan çağrılarda CPU maliyeti sıfır.
        """
        if cls._schema_cache is None:
            schema = cls.model_json_schema()
            cls._schema_cache = json.dumps(schema, indent=2, ensure_ascii=False)
        return cls._schema_cache


# ---------------------------------------------------------------------------
# Batch İşleme Modeli - Toplu AI çıktısı için wrapper
# ---------------------------------------------------------------------------

class AIBatchResult(BaseModel):
    """
    Bir batch AI çağrısının sonucunu sarmalayan model.
    Başarılı ve başarısız kayıtları ayrı tutar.
    Maliyet takibi için token_usage alanı eklenmiştir.
    """

    batch_id: str = Field(
        ...,
        description="Batch işlem kimliği (UUID prefix).",
        examples=["65f6ca95", "91fd188d"],
    )
    total_input: int = Field(
        ...,
        ge=0,
        description="Pipeline'a giren toplam ham kayıt sayısı.",
        examples=[20, 100],
    )
    successful: list[AIAnalyzedReview] = Field(
        default_factory=list,
        description="Başarıyla analiz edilen ve doğrulanan AIAnalyzedReview listesi.",
    )
    failed_ids: list[str] = Field(
        default_factory=list,
        description="Analizi başarısız olan kayıt source_id'leri.",
        examples=[["comment_5", "comment_12"]],
    )
    processing_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Toplam AI işleme süresi (saniye).",
        examples=[2.34, 12.7],
    )
    model_used: str = Field(
        default="mock",
        description="Bu batch için kullanılan AI model adı.",
        examples=["gpt-4o-mini", "claude-3-5-haiku-20241022", "mock-v1.0-deterministic"],
    )
    total_tokens_used: int = Field(
        default=0,
        ge=0,
        description="Bu batch'te harcanan toplam token sayısı (maliyet takibi).",
        examples=[4500, 0],
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Tahmini API maliyeti (USD). Mock modda her zaman 0.0.",
        examples=[0.0045, 0.0],
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Batch işleminin tamamlandığı zaman (UTC).",
    )

    @property
    def success_rate(self) -> float:
        """Başarı oranını hesapla (0.0 - 1.0)."""
        if self.total_input == 0:
            return 0.0
        return len(self.successful) / self.total_input

    @property
    def failed_count(self) -> int:
        """Başarısız kayıt sayısı."""
        return len(self.failed_ids)

    @property
    def avg_processing_time_per_record(self) -> float:
        """Kayıt başına ortalama işleme süresi (saniye)."""
        if self.total_input == 0:
            return 0.0
        return round(self.processing_time_seconds / self.total_input, 4)


# ---------------------------------------------------------------------------
# Analitik Özet Modeli - Analytics katmanının çıktısı
# ---------------------------------------------------------------------------

class AnalyticsSummary(BaseModel):
    """
    pandas analizinin sonuçlarını taşıyan özet rapor modeli.
    Dashboard, executive summary ve alert sistemi için kullanılır.

    Enterprise Eklentileri:
      - time_series_data  : Gün/hafta bazlı sentiment trendi (görselleştirme)
      - anomaly_detected  : İstatistiksel anomali tespit flag'i (alert sistemi)
      - anomaly_details   : Hangi periyotta, hangi metrik, z-score kaç
    """

    report_id: str = Field(
        ...,
        description="Rapor benzersiz kimliği (UUID prefix).",
        examples=["ac68d9cc-25e", "9cf11d5a-888"],
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Raporun üretildiği zaman (UTC, timezone-aware).",
    )

    # --- Genel İstatistikler ---
    total_reviews_analyzed: int = Field(
        0, ge=0,
        description="Temizleme sonrası analize giren toplam kayıt sayısı.",
        examples=[150],
    )
    high_quality_reviews: int = Field(
        0, ge=0,
        description="data_quality='high' olan kayıt sayısı.",
        examples=[120],
    )
    spam_excluded: int = Field(
        0, ge=0,
        description="Spam/sahte olarak filtrelenen kayıt sayısı.",
        examples=[5],
    )

    # --- Duygu Metrikleri ---
    sentiment_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Her sentiment label için kayıt sayısı.",
        examples=[{"positive": 80, "negative": 30, "neutral": 25, "mixed": 15}],
    )
    average_sentiment_score: float = Field(
        0.0,
        description="Ortalama sentiment skoru (-1.0 ile +1.0).",
        examples=[0.23],
    )
    average_rating: Optional[float] = Field(
        None,
        description="Ortalama müşteri puanı (1.0-5.0). Veri yoksa null.",
        examples=[3.87, None],
    )
    average_confidence: float = Field(
        0.0,
        description="Ortalama AI güven skoru (0.0-1.0).",
        examples=[0.84],
    )

    # --- Kategori Dağılımı ---
    category_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Ürün kategorisi bazında kayıt sayısı.",
        examples=[{"electronics": 60, "books": 30}],
    )

    # --- Aksiyon Metrikleri ---
    action_required_count: int = Field(
        0, ge=0,
        description="Customer support aksiyonu gerektiren kayıt sayısı.",
        examples=[12],
    )
    critical_urgency_count: int = Field(
        0, ge=0,
        description="CRITICAL öncelikli kayıt sayısı.",
        examples=[2],
    )
    high_urgency_count: int = Field(
        0, ge=0,
        description="HIGH öncelikli kayıt sayısı (24 saat içinde aksiyon).",
        examples=[8],
    )

    # --- Konu ve Rakip Analizi ---
    top_topics: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top-10 konu: {topic, count, avg_sentiment}.",
        examples=[[{"topic": "battery life", "count": 23, "avg_sentiment": 0.65}]],
    )
    competitor_mentions: dict[str, int] = Field(
        default_factory=dict,
        description="Rakip marka geçme sayısı.",
        examples=[{"CompetitorX": 5, "BrandY": 3}],
    )

    # --- Veri Kalitesi ve Dil ---
    data_quality_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Veri kalitesi seviyesi dağılımı.",
        examples=[{"high": 80, "medium": 50, "low": 20}],
    )
    language_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Dil dağılımı (ISO 639-1).",
        examples=[{"en": 120, "tr": 20}],
    )

    # --- Zaman Serisi (Time-Series Analytics) ---
    time_series_data: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Gün/hafta bazlı aggregated metrikler. "
            "Her item: {period, avg_sentiment, review_count, avg_rating, negative_rate}."
        ),
        examples=[[
            {"period": "2024-03-15", "avg_sentiment": 0.45,
             "review_count": 12, "avg_rating": 4.1, "negative_rate": 0.08},
        ]],
    )

    # --- Anomali Tespiti ---
    anomaly_detected: bool = Field(
        default=False,
        description=(
            "Zaman serisinde istatistiksel anomali (z-score > threshold) tespit edildi mi? "
            "True ise anomaly_details alanında detay bulunur; alert mekanizması tetiklenmelidir."
        ),
        examples=[True, False],
    )
    anomaly_details: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Tespit edilen anomali olayları. "
            "Her item: {period, metric, value, z_score, severity}."
        ),
        examples=[[
            {"period": "2024-03-16", "metric": "negative_rate",
             "value": 0.75, "z_score": 3.2, "severity": "high"},
        ]],
    )
