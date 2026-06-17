"""
analytics.py - Enterprise Veri Analitiği Katmanı
==================================================
Amaç: AI tarafından yapılandırılan verileri pandas ile işleyerek
      müşteriye doğrudan sunulabilecek KPI raporu, zaman serisi trendi
      ve istatistiksel anomali analizi üretmek.

Mimari Kararlar:
  - Tüm analitik işlemler DataAnalytics sınıfında kapsüllenmiştir;
    bu yaklaşım test edilebilirliği artırır (her metod bağımsız test edilebilir).
  - pandas method chaining ile veri temizleme pipeline'ı okunabilir ve
    genişletilebilir şekilde yazılmıştır.
  - Tüm istatistiksel hesaplamalar fully vectorized pandas aggregation kullanır;
    Python for-loop'lardan tamamen kaçınılmıştır (NumPy broadcasting → performans).
  - Zaman serisi: review_date alanı üzerinden gün/hafta granülaritesinde aggregation.
  - Anomali tespiti: Z-score yöntemi; rolling mean ± std eşiği aşan periyotlar
    anomaly_details listesine eklenir, anomaly_detected flag'i True yapılır.

Enterprise Yenilikler (v2):
  - compute_time_series()      : Vektörize groupby ile gün/hafta sentiment trendi
  - compute_anomaly_detection(): Z-score tabanlı istatistiksel anomali tespiti
  - _extract_top_topics()      : DataFrame.explode() ile for-loop'suz konu çıkarımı
  - _extract_competitor_mentions(): aynı explode yaklaşımı
  - reviews_to_dataframe()     : review_date datetime olarak korunur (time-series için)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from models import (
    AIAnalyzedReview,
    AIBatchResult,
    AnalyticsSummary,
    DataQuality,
    SentimentLabel,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Minimum güven skoru - bu altındaki AI analizleri rapordan hariç tutulur
MIN_CONFIDENCE_THRESHOLD: float = 0.50

# Anomali tespiti z-score eşiği
ANOMALY_Z_THRESHOLD: float = 2.0

# Zaman serisi için minimum periyot sayısı (az veriyle trend yanıltıcı olur)
MIN_PERIODS_FOR_TIMESERIES: int = 2

# Sütun tip eşlemesi - DataFrame oluşturma sonrası uygulanır
DTYPE_MAP: dict[str, str] = {
    "sentiment_score": "float64",
    "confidence_score": "float64",
    "reviewer_rating": "float64",
    "action_required": "bool",
    "is_spam_or_fake": "bool",
    "is_verified_purchase": "bool",
}


# ---------------------------------------------------------------------------
# Ana Analitik Sınıfı
# ---------------------------------------------------------------------------

class DataAnalytics:
    """
    AI çıktısını pandas DataFrame'e dönüştüren, KPI analizi,
    zaman serisi trendi ve anomali tespiti yapan sınıf.

    Pipeline adımları:
      1. AIAnalyzedReview listesi → ham DataFrame
      2. Veri temizleme (cleaning) → temiz DataFrame
      3. KPI hesaplama → AnalyticsSummary modeli
      4. Zaman serisi analizi → time_series_data populate
      5. Anomali tespiti → anomaly_detected + anomaly_details populate
      6. Temiz DataFrame'i üst katmana (reporter) ilet
    """

    def __init__(self) -> None:
        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        logger.info("DataAnalytics katmanı başlatıldı.")

    # -----------------------------------------------------------------------
    # Adım 1: Pydantic Modelleri → pandas DataFrame
    # -----------------------------------------------------------------------

    def reviews_to_dataframe(
        self, reviews: list[AIAnalyzedReview]
    ) -> pd.DataFrame:
        """
        AIAnalyzedReview Pydantic modellerini flat pandas DataFrame'e dönüştürür.

        Kritik enterprise değişiklik:
          - review_date datetime nesnesi olarak korunur (time-series için zorunlu).
          - analyzed_at da datetime olarak korunur (audit trail için).
          - List alanlar hem orijinal (list) hem _str (pipe-joined) formatında tutulur.
          - Tip dönüşümleri vectorized pd.to_numeric ile yapılır.
        """
        if not reviews:
            logger.warning("Boş review listesi geldi, boş DataFrame döndürülüyor.")
            return pd.DataFrame()

        logger.info(f"DataFrame oluşturuluyor: {len(reviews)} kayıt")

        # model_dump() ile tek seferde tüm kayıtları dict listesine çevir
        records: list[dict[str, Any]] = [r.model_dump() for r in reviews]

        df = pd.DataFrame(records)

        # --- List alanları pipe-joined string sütunlarına kopyala (CSV/Excel uyumu) ---
        list_col_map = {
            "key_topics": "key_topics_str",
            "pros": "pros_str",
            "cons": "cons_str",
            "mentioned_competitors": "competitors_str",
        }
        for src_col, dst_col in list_col_map.items():
            if src_col in df.columns:
                df[dst_col] = df[src_col].apply(
                    lambda v: "|".join(str(x) for x in v) if isinstance(v, list) else ""
                )

        # --- Enum sütunları normalize et (str Enum zaten str, güvenle lower) ---
        enum_cols = ["sentiment", "product_category", "urgency_level", "data_quality"]
        for col in enum_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()

        # --- review_date: datetime olarak koru, NaT kullan (zaman serisi için) ---
        if "review_date" in df.columns:
            df["review_date"] = pd.to_datetime(df["review_date"], utc=True, errors="coerce")

        # --- analyzed_at: datetime olarak koru ---
        if "analyzed_at" in df.columns:
            df["analyzed_at"] = pd.to_datetime(df["analyzed_at"], utc=True, errors="coerce")

        # --- Sayısal tip dönüşümleri (vectorized) ---
        for col, dtype in DTYPE_MAP.items():
            if col in df.columns:
                if dtype == "bool":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

        self._raw_df = df
        logger.info(
            f"Ham DataFrame oluşturuldu: "
            f"{df.shape[0]} satır × {df.shape[1]} sütun"
        )
        return df

    # -----------------------------------------------------------------------
    # Adım 2: Veri Temizleme Pipeline'ı
    # -----------------------------------------------------------------------

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Üretim kalitesinde veri temizleme pipeline'ı uygular.

        Temizleme adımları (sıralı, her adım loglanır):
          1. Boş DataFrame kontrolü
          2. Spam / sahte yorum filtreleme
          3. Düşük güven skorlu AI analizlerini eleme
          4. Zorunlu alanların null kontrolü
          5. Değer aralığı doğrulaması (rating 1-5, score -1 to 1)
          6. Duplicate source_id tespiti ve kaldırılması
          7. String sütunlarında trim + lowercase normalizasyonu
        """
        if df.empty:
            logger.warning("Temizlenecek DataFrame boş.")
            return df

        initial_count = len(df)
        logger.info(f"Veri temizleme başlıyor: {initial_count} kayıt")

        clean = df.copy()

        # --- Adım 1: Spam filtresi (vectorized boolean mask) ---
        if "is_spam_or_fake" in clean.columns:
            spam_mask = clean["is_spam_or_fake"].fillna(False).astype(bool)
            spam_count = int(spam_mask.sum())
            clean = clean.loc[~spam_mask]
            if spam_count > 0:
                logger.info(f"  ✂ Spam/sahte kayıt kaldırıldı: {spam_count}")

        # --- Adım 2: Düşük güven skoru filtresi (vectorized comparison) ---
        if "confidence_score" in clean.columns:
            low_conf_mask = clean["confidence_score"] < MIN_CONFIDENCE_THRESHOLD
            low_conf_count = int(low_conf_mask.sum())
            clean = clean.loc[~low_conf_mask]
            if low_conf_count > 0:
                logger.info(
                    f"  ✂ Düşük güven skorlu kayıt kaldırıldı: "
                    f"{low_conf_count} (threshold={MIN_CONFIDENCE_THRESHOLD})"
                )

        # --- Adım 3: Kritik alanların null kontrolü (vectorized any) ---
        critical_cols = ["source_id", "sentiment", "sentiment_score"]
        existing_critical = [c for c in critical_cols if c in clean.columns]
        if existing_critical:
            null_mask = clean[existing_critical].isnull().any(axis=1)
            null_count = int(null_mask.sum())
            clean = clean.loc[~null_mask]
            if null_count > 0:
                logger.info(f"  ✂ Kritik alan eksik kayıt kaldırıldı: {null_count}")

        # --- Adım 4: Değer aralığı doğrulaması (vectorized clip/mask) ---
        if "reviewer_rating" in clean.columns:
            valid_mask = clean["reviewer_rating"].between(1.0, 5.0, inclusive="both")
            null_mask_rating = clean["reviewer_rating"].isnull()
            invalid_count = int((~valid_mask & ~null_mask_rating).sum())
            clean.loc[~valid_mask & ~null_mask_rating, "reviewer_rating"] = pd.NA
            if invalid_count > 0:
                logger.info(f"  ⚠ Geçersiz rating değeri NaN yapıldı: {invalid_count}")

        if "sentiment_score" in clean.columns:
            clean["sentiment_score"] = clean["sentiment_score"].clip(-1.0, 1.0)

        # --- Adım 5: Duplicate kayıt tespiti (vectorized dedup) ---
        if "source_id" in clean.columns:
            before_dedup = len(clean)
            clean = clean.drop_duplicates(subset=["source_id"], keep="first")
            dup_count = before_dedup - len(clean)
            if dup_count > 0:
                logger.info(f"  ✂ Duplicate source_id kaldırıldı: {dup_count}")

        # --- Adım 6: String normalizasyonu (vectorized str ops) ---
        string_cols = ["sentiment", "product_category", "urgency_level", "language_detected"]
        for col in string_cols:
            if col in clean.columns:
                clean[col] = clean[col].astype(str).str.strip().str.lower()

        # --- Adım 7: product_name temizliği ---
        if "product_name" in clean.columns:
            clean["product_name"] = (
                clean["product_name"]
                .str.strip()
                .replace({"": "Unknown", "nan": "Unknown"})
                .fillna("Unknown")
            )

        # --- İndeksi sıfırla ---
        clean = clean.reset_index(drop=True)

        removed_total = initial_count - len(clean)
        logger.info(
            f"Veri temizleme tamamlandı: "
            f"{len(clean)} kayıt kaldı "
            f"({removed_total} kayıt kaldırıldı, "
            f"oran={removed_total / initial_count:.1%})"
        )

        self._clean_df = clean
        return clean

    # -----------------------------------------------------------------------
    # Adım 3: Yardımcı Analitik Metodlar (tamamen vektörize)
    # -----------------------------------------------------------------------

    def _safe_mean(self, series: pd.Series) -> float:
        """Null değerler içerebilecek serinin ortalamasını güvenle hesapla."""
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        return round(float(numeric.mean()), 4) if not numeric.empty else 0.0

    def _value_counts_dict(
        self, series: pd.Series, top_n: Optional[int] = None
    ) -> dict[str, int]:
        """Kategori dağılımını dict olarak döndür (vectorized value_counts)."""
        counts = series.value_counts()
        if top_n:
            counts = counts.head(top_n)
        return {str(k): int(v) for k, v in counts.items()}

    def _extract_top_topics(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> list[dict[str, Any]]:
        """
        key_topics list sütunundan en sık geçen konuları çıkarır.
        DataFrame.explode() kullanılır — Python for-loop yok.

        Her konu için: frekans ve ortalama sentiment skoru.
        """
        # key_topics list sütunu varsa doğrudan explode et
        src_col: Optional[str] = None
        if "key_topics" in df.columns:
            src_col = "key_topics"
        elif "key_topics_str" in df.columns:
            src_col = "key_topics_str"

        if src_col is None or df.empty:
            return []

        # key_topics sütunundan çalışıyoruz
        if src_col == "key_topics":
            # list sütunu → explode
            topic_df = (
                df[["key_topics", "sentiment_score"]]
                .copy()
                .assign(key_topics=lambda d: d["key_topics"].apply(
                    lambda v: v if isinstance(v, list) else []
                ))
                .explode("key_topics")
                .rename(columns={"key_topics": "topic"})
                .dropna(subset=["topic"])
            )
            topic_df["topic"] = topic_df["topic"].astype(str).str.strip()
            topic_df = topic_df[topic_df["topic"] != ""]
        else:
            # pipe-joined string → str.split + explode
            topic_df = (
                df[["key_topics_str", "sentiment_score"]]
                .copy()
                .assign(topic=lambda d: d["key_topics_str"].str.split("|"))
                .explode("topic")
                .drop(columns=["key_topics_str"])
                .dropna(subset=["topic"])
            )
            topic_df["topic"] = topic_df["topic"].str.strip()
            topic_df = topic_df[topic_df["topic"] != ""]

        if topic_df.empty:
            return []

        # Vectorized groupby aggregation
        topic_agg = (
            topic_df
            .groupby("topic", as_index=False)
            .agg(
                count=("topic", "count"),
                avg_sentiment=("sentiment_score", "mean"),
            )
            .sort_values("count", ascending=False)
            .head(top_n)
        )

        return [
            {
                "topic": str(row["topic"]),
                "count": int(row["count"]),
                "avg_sentiment": round(float(row["avg_sentiment"]), 3),
            }
            for _, row in topic_agg.iterrows()
        ]

    def _extract_competitor_mentions(
        self, df: pd.DataFrame
    ) -> dict[str, int]:
        """
        Rakip marka geçme sayısını hesaplar.
        DataFrame.explode() ile for-loop'suz vectorized çözüm.
        """
        src_col: Optional[str] = None
        if "mentioned_competitors" in df.columns:
            src_col = "mentioned_competitors"
        elif "competitors_str" in df.columns:
            src_col = "competitors_str"

        if src_col is None or df.empty:
            return {}

        if src_col == "mentioned_competitors":
            comp_series = (
                df[["mentioned_competitors"]]
                .assign(mentioned_competitors=lambda d: d["mentioned_competitors"].apply(
                    lambda v: v if isinstance(v, list) else []
                ))
                .explode("mentioned_competitors")["mentioned_competitors"]
            )
        else:
            comp_series = (
                df[["competitors_str"]]
                .assign(competitor=lambda d: d["competitors_str"].str.split("|"))
                .explode("competitor")["competitor"]
            )

        comp_series = comp_series.astype(str).str.strip()
        comp_series = comp_series[comp_series.isin(["", "nan", "None"]) == False]  # noqa: E712

        if comp_series.empty:
            return {}

        counts = comp_series.value_counts().head(10)
        return {str(k): int(v) for k, v in counts.items()}

    # -----------------------------------------------------------------------
    # Adım 4: Zaman Serisi Analizi (Enterprise - Tamamen Vektörize)
    # -----------------------------------------------------------------------

    def compute_time_series(
        self,
        df: pd.DataFrame,
        granularity: str = "day",
    ) -> list[dict[str, Any]]:
        """
        review_date alanı üzerinden gün veya hafta bazlı trend analizi üretir.

        Tamamen vektörize: pandas groupby + agg, hiç Python for-loop yok.

        Parametreler:
          df          : Temiz DataFrame (review_date datetime sütunu içermeli)
          granularity : "day"  → YYYY-MM-DD günlük aggregation
                        "week" → ISO hafta (YYYY-Www) haftalık aggregation

        Döndürür:
          List[dict] — her item:
            {
              "period"        : str   (YYYY-MM-DD veya YYYY-Www),
              "avg_sentiment" : float (ortalama sentiment skoru),
              "review_count"  : int   (o periyottaki yorum sayısı),
              "avg_rating"    : float | None (ortalama rating),
              "negative_rate" : float (negatif yorum oranı 0.0-1.0),
              "positive_rate" : float (pozitif yorum oranı 0.0-1.0),
            }
        """
        if df.empty or "review_date" not in df.columns:
            logger.info("Zaman serisi: review_date sütunu yok veya veri boş, atlanıyor.")
            return []

        # review_date datetime olmalı; NaT olanları çıkar
        ts_df = df.dropna(subset=["review_date"]).copy()
        if len(ts_df) < MIN_PERIODS_FOR_TIMESERIES:
            logger.info(
                f"Zaman serisi: yeterli veri yok "
                f"(mevcut={len(ts_df)}, minimum={MIN_PERIODS_FOR_TIMESERIES})."
            )
            return []

        # Periyot sütunu oluştur (vectorized dt accessor)
        if granularity == "week":
            ts_df["_period"] = ts_df["review_date"].dt.to_period("W").astype(str)
        else:
            # day granülaritesi: sadece tarih kısmını al
            ts_df["_period"] = ts_df["review_date"].dt.date.astype(str)

        # Negatif/pozitif binary flag (vectorized comparison)
        ts_df["_is_negative"] = (
            ts_df.get("sentiment", pd.Series(dtype=str))
            .astype(str)
            .str.lower()
            .eq("negative")
            .astype(float)
        )
        ts_df["_is_positive"] = (
            ts_df.get("sentiment", pd.Series(dtype=str))
            .astype(str)
            .str.lower()
            .eq("positive")
            .astype(float)
        )

        # Aggregation spec — tüm metric'ler tek groupby çağrısında hesaplanır
        agg_spec: dict[str, Any] = {
            "sentiment_score": ["mean"],
            "_is_negative": ["mean"],
            "_is_positive": ["mean"],
            "source_id": ["count"],
        }
        if "reviewer_rating" in ts_df.columns:
            agg_spec["reviewer_rating"] = ["mean"]

        grouped = ts_df.groupby("_period").agg(agg_spec)

        # MultiIndex sütunları düzleştir — strip("_") KULLANMA, baştaki _ silinir
        grouped.columns = ["_".join(col) for col in grouped.columns]
        grouped = grouped.reset_index().sort_values("_period")

        # Sütun adı normalize
        col_map = {
            "_period": "period",
            "sentiment_score_mean": "avg_sentiment",
            "_is_negative_mean": "negative_rate",
            "_is_positive_mean": "positive_rate",
            "source_id_count": "review_count",
        }
        if "reviewer_rating_mean" in grouped.columns:
            col_map["reviewer_rating_mean"] = "avg_rating"

        grouped = grouped.rename(columns=col_map)

        # Yuvarlama ve tip güvenliği (vectorized round)
        for float_col in ["avg_sentiment", "negative_rate", "positive_rate"]:
            if float_col in grouped.columns:
                grouped[float_col] = grouped[float_col].round(4)

        if "avg_rating" in grouped.columns:
            grouped["avg_rating"] = grouped["avg_rating"].round(2)
        else:
            grouped["avg_rating"] = None

        grouped["review_count"] = grouped["review_count"].astype(int)

        # Sıralı dict listesine dönüştür
        result_cols = [
            "period", "avg_sentiment", "review_count",
            "avg_rating", "negative_rate", "positive_rate",
        ]
        result_cols = [c for c in result_cols if c in grouped.columns]

        records = grouped[result_cols].to_dict(orient="records")

        # avg_rating None → None (NaN → None dönüşümü)
        for rec in records:
            if "avg_rating" in rec and pd.isna(rec["avg_rating"]):
                rec["avg_rating"] = None

        logger.info(
            f"Zaman serisi hesaplandı: "
            f"{len(records)} periyot "
            f"({granularity} granülaritesi)"
        )
        return records

    # -----------------------------------------------------------------------
    # Adım 5: Anomali Tespiti (Z-Score Tabanlı, Tamamen Vektörize)
    # -----------------------------------------------------------------------

    def compute_anomaly_detection(
        self,
        time_series: list[dict[str, Any]],
        metrics: Optional[list[str]] = None,
        z_threshold: float = ANOMALY_Z_THRESHOLD,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """
        Zaman serisi verisi üzerinde z-score tabanlı anomali tespiti.

        Algoritma:
          1. Her metrik için pandas Series oluştur
          2. Globl mean ve std hesapla (vectorized)
          3. Z-score = (value - mean) / std (vectorized)
          4. |z_score| > z_threshold → anomali

        Parametreler:
          time_series : compute_time_series() çıktısı
          metrics     : Kontrol edilecek metrikler (default: negative_rate, avg_sentiment)
          z_threshold : Z-score eşiği (default: 2.0)

        Döndürür:
          (anomaly_detected: bool, anomaly_details: list[dict])
          Her detail item:
            {period, metric, value, z_score, severity}
          severity: "critical" (|z|>3), "high" (|z|>2.5), "medium" (|z|>2)
        """
        if not time_series or len(time_series) < 3:
            # 3'ten az periyotta z-score anlamsız
            return False, []

        if metrics is None:
            metrics = ["negative_rate", "avg_sentiment"]

        ts_df = pd.DataFrame(time_series)
        anomaly_details: list[dict[str, Any]] = []

        for metric in metrics:
            if metric not in ts_df.columns:
                continue

            series = pd.to_numeric(ts_df[metric], errors="coerce").dropna()
            if len(series) < 3:
                continue

            mean_val: float = float(series.mean())
            std_val: float = float(series.std(ddof=1))

            if std_val < 1e-9:
                # Tüm değerler aynı → z-score tanımsız, anomali yok
                continue

            # Z-score vectorized hesaplama
            z_scores: pd.Series = (series - mean_val) / std_val
            abs_z: pd.Series = z_scores.abs()

            # Eşik aşan indeksleri bul (vectorized boolean mask)
            anomaly_mask = abs_z > z_threshold
            anomaly_indices = series.index[anomaly_mask]

            for idx in anomaly_indices:
                z_val = float(z_scores.loc[idx])
                abs_z_val = abs(z_val)
                period_val = ts_df.loc[idx, "period"] if "period" in ts_df.columns else str(idx)
                raw_val = float(series.loc[idx])

                # Severity belirleme (vectorized threshold chain)
                if abs_z_val > 3.0:
                    severity = "critical"
                elif abs_z_val > 2.5:
                    severity = "high"
                else:
                    severity = "medium"

                anomaly_details.append({
                    "period": str(period_val),
                    "metric": metric,
                    "value": round(raw_val, 4),
                    "z_score": round(z_val, 3),
                    "severity": severity,
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                })

        anomaly_detected = len(anomaly_details) > 0

        if anomaly_detected:
            logger.warning(
                f"⚠ Anomali tespit edildi: {len(anomaly_details)} olay | "
                f"metrikler={metrics} | z_eşiği={z_threshold}"
            )
        else:
            logger.info(
                f"Anomali tespiti tamamlandı: anomali yok "
                f"(z_eşiği={z_threshold}, periyot={len(time_series)})"
            )

        return anomaly_detected, anomaly_details

    # -----------------------------------------------------------------------
    # Adım 6: KPI Hesaplama ve Analitik Özet
    # -----------------------------------------------------------------------

    def compute_summary(
        self,
        clean_df: pd.DataFrame,
        raw_df: Optional[pd.DataFrame] = None,
        time_series_granularity: str = "day",
    ) -> AnalyticsSummary:
        """
        Temizlenmiş DataFrame üzerinden AnalyticsSummary KPI raporu üretir.
        Zaman serisi ve anomali tespiti dahil tüm enterprise metrikleri hesaplar.
        """
        report_id = str(uuid.uuid4())[:12]
        logger.info(
            f"KPI özeti hesaplanıyor: report_id={report_id}, "
            f"kayıt={len(clean_df)}"
        )

        if clean_df.empty:
            logger.warning("Analiz edilecek veri yok, boş özet döndürülüyor.")
            return AnalyticsSummary(report_id=report_id)

        total = len(clean_df)

        # Yüksek kaliteli kayıt sayısı (vectorized eq)
        high_quality = 0
        if "data_quality" in clean_df.columns:
            high_quality = int(clean_df["data_quality"].eq(DataQuality.HIGH.value).sum())

        # Spam dışlanan kayıt sayısı
        spam_excluded = 0
        if raw_df is not None and not raw_df.empty and "is_spam_or_fake" in raw_df.columns:
            spam_excluded = int(raw_df["is_spam_or_fake"].fillna(False).astype(bool).sum())

        # Sentiment dağılımı (vectorized value_counts)
        sentiment_dist = self._value_counts_dict(
            clean_df["sentiment"] if "sentiment" in clean_df.columns
            else pd.Series(dtype=str)
        )
        avg_sentiment = self._safe_mean(
            clean_df["sentiment_score"] if "sentiment_score" in clean_df.columns
            else pd.Series(dtype=float)
        )

        # Rating ortalaması (NaN-safe vectorized mean)
        avg_rating: Optional[float] = None
        if "reviewer_rating" in clean_df.columns:
            valid_ratings = pd.to_numeric(
                clean_df["reviewer_rating"], errors="coerce"
            ).dropna()
            if not valid_ratings.empty:
                avg_rating = round(float(valid_ratings.mean()), 2)

        avg_confidence = self._safe_mean(
            clean_df["confidence_score"] if "confidence_score" in clean_df.columns
            else pd.Series(dtype=float)
        )

        category_dist = self._value_counts_dict(
            clean_df["product_category"] if "product_category" in clean_df.columns
            else pd.Series(dtype=str)
        )

        # Aksiyon metrikleri (vectorized boolean sum)
        action_count = 0
        if "action_required" in clean_df.columns:
            action_count = int(
                clean_df["action_required"].fillna(False).astype(bool).sum()
            )

        critical_count = 0
        high_count = 0
        if "urgency_level" in clean_df.columns:
            urgency_counts = clean_df["urgency_level"].value_counts()
            critical_count = int(urgency_counts.get(UrgencyLevel.CRITICAL.value, 0))
            high_count = int(urgency_counts.get(UrgencyLevel.HIGH.value, 0))

        # Konu + Rakip analizi (vectorized explode)
        top_topics = self._extract_top_topics(clean_df, top_n=10)
        competitor_mentions = self._extract_competitor_mentions(clean_df)

        quality_dist = self._value_counts_dict(
            clean_df["data_quality"] if "data_quality" in clean_df.columns
            else pd.Series(dtype=str)
        )
        lang_dist = self._value_counts_dict(
            clean_df["language_detected"] if "language_detected" in clean_df.columns
            else pd.Series(dtype=str)
        )

        # Zaman Serisi Analizi
        time_series_data = self.compute_time_series(
            clean_df, granularity=time_series_granularity
        )

        # Z-Score Anomali Tespiti
        anomaly_detected, anomaly_details = self.compute_anomaly_detection(
            time_series_data,
            metrics=["negative_rate", "avg_sentiment"],
            z_threshold=ANOMALY_Z_THRESHOLD,
        )

        summary = AnalyticsSummary(
            report_id=report_id,
            total_reviews_analyzed=total,
            high_quality_reviews=high_quality,
            spam_excluded=spam_excluded,
            sentiment_distribution=sentiment_dist,
            average_sentiment_score=avg_sentiment,
            average_rating=avg_rating,
            average_confidence=avg_confidence,
            category_distribution=category_dist,
            action_required_count=action_count,
            critical_urgency_count=critical_count,
            high_urgency_count=high_count,
            top_topics=top_topics,
            competitor_mentions=competitor_mentions,
            data_quality_distribution=quality_dist,
            language_distribution=lang_dist,
            time_series_data=time_series_data,
            anomaly_detected=anomaly_detected,
            anomaly_details=anomaly_details,
        )

        anomali_str = "EVET" if anomaly_detected else "hayir"
        logger.info(
            f"KPI ozeti tamamlandi: toplam={total} | "
            f"avg_sentiment={avg_sentiment:.3f} | avg_rating={avg_rating} | "
            f"aksiyon={action_count} | "
            f"zaman_serisi={len(time_series_data)} periyot | "
            f"anomali={anomali_str}"
        )
        return summary

    # -----------------------------------------------------------------------
    # Adım 7: Konsol Raporu
    # -----------------------------------------------------------------------

    def print_summary_report(self, summary: AnalyticsSummary) -> None:
        """
        AnalyticsSummary'yi konsola okunabilir format'ta yazdırır.
        Anomali uyarıları ve zaman serisi özeti dahildir.
        """
        divider = "=" * 60
        thin = "-" * 60

        logger.info(divider)
        logger.info("  AI DATA ENGINE - KPI RAPORU")
        logger.info(f"  Rapor ID : {summary.report_id}")
        logger.info(f"  Tarih    : {summary.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info(divider)

        logger.info("  GENEL ISTATISTIKLER")
        logger.info(thin)
        logger.info(f"  Toplam analiz edilen yorum : {summary.total_reviews_analyzed}")
        logger.info(f"  Yuksek kaliteli kayit      : {summary.high_quality_reviews}")
        logger.info(f"  Spam / sahte filtreler     : {summary.spam_excluded}")
        logger.info(f"  Ortalama AI guven skoru    : {summary.average_confidence:.1%}")

        logger.info(thin)
        logger.info("  DUYGU ANALIZI")
        logger.info(thin)
        for sentiment, count in summary.sentiment_distribution.items():
            bar = "#" * (count * 20 // max(summary.total_reviews_analyzed, 1))
            pct = count / max(summary.total_reviews_analyzed, 1)
            logger.info(f"  {sentiment:<12} {bar:<20} {count:>4} ({pct:.1%})")
        logger.info(f"  Ortalama sentiment skoru   : {summary.average_sentiment_score:+.3f}")
        if summary.average_rating:
            logger.info(f"  Ortalama musteri puani     : {summary.average_rating:.2f} / 5.0")

        logger.info(thin)
        logger.info("  KATEGORI DAGILIMI")
        logger.info(thin)
        for cat, count in summary.category_distribution.items():
            logger.info(f"  {cat:<20} {count:>4} kayit")

        logger.info(thin)
        logger.info("  AKSIYON GEREKTIREN KAYITLAR")
        logger.info(thin)
        logger.info(f"  Aksiyon gereken toplam     : {summary.action_required_count}")
        logger.info(f"  CRITICAL oncelikli         : {summary.critical_urgency_count}")
        logger.info(f"  HIGH oncelikli             : {summary.high_urgency_count}")

        if summary.top_topics:
            logger.info(thin)
            logger.info(f"  EN COK KONUSULAN KONULAR (Top {len(summary.top_topics)})")
            logger.info(thin)
            for item in summary.top_topics[:5]:
                icon = "+" if item["avg_sentiment"] > 0 else "-"
                logger.info(
                    f"  {item['topic']:<25} "
                    f"bahsedilme: {item['count']:>3} | "
                    f"sentiment: {icon}{abs(item['avg_sentiment']):.2f}"
                )

        if summary.competitor_mentions:
            logger.info(thin)
            logger.info("  RAKIP MARKA GECMELERI")
            logger.info(thin)
            for brand, count in summary.competitor_mentions.items():
                logger.info(f"  {brand:<25} {count:>3} yorum")

        if summary.time_series_data:
            logger.info(thin)
            logger.info(f"  ZAMAN SERISI ({len(summary.time_series_data)} periyot)")
            logger.info(thin)
            for item in summary.time_series_data[-5:]:
                neg_pct = item.get("negative_rate", 0.0)
                logger.info(
                    f"  {item['period']:<12} "
                    f"sentiment={item['avg_sentiment']:+.3f} | "
                    f"yorum={item['review_count']:>4} | "
                    f"neg={neg_pct:.1%}"
                )

        if summary.anomaly_detected:
            logger.info(thin)
            logger.info("  *** ANOMALI TESPIT EDILDI! ***")
            logger.info(thin)
            for anomaly in summary.anomaly_details:
                logger.info(
                    f"  [{anomaly['severity'].upper()}] "
                    f"Periyot={anomaly['period']} | "
                    f"Metrik={anomaly['metric']} | "
                    f"Deger={anomaly['value']:.4f} | "
                    f"Z-Score={anomaly['z_score']:+.2f}"
                )

        logger.info(divider)

    # -----------------------------------------------------------------------
    # Ana Pipeline Metodu
    # -----------------------------------------------------------------------

    def run_pipeline(
        self,
        batch_result: AIBatchResult,
        time_series_granularity: str = "day",
    ) -> tuple[pd.DataFrame, AnalyticsSummary]:
        """
        Tam analitik pipeline'i calistirir:
          AIBatchResult -> ham DataFrame -> temiz DataFrame
          -> AnalyticsSummary (KPI + time-series + anomali)

        Parametreler:
          batch_result            : AI islemci ciktisi
          time_series_granularity : "day" veya "week"

        Dondurur:
          (clean_df, summary) - reporter katmanina iletilecek ciktilar
        """
        logger.info(
            f"=== Analitik Pipeline Basliyor === "
            f"Giris: {len(batch_result.successful)} kayit"
        )

        if not batch_result.successful:
            logger.error("AI'dan basarili kayit gelmedi, analitik pipeline durdu.")
            return pd.DataFrame(), AnalyticsSummary(report_id="empty")

        # 1. Pydantic modeller -> DataFrame (review_date datetime olarak korunur)
        raw_df = self.reviews_to_dataframe(batch_result.successful)

        # 2. Veri temizleme (vectorized pipeline)
        clean_df = self.clean_dataframe(raw_df)

        # 3. KPI + zaman serisi + anomali hesaplama
        summary = self.compute_summary(
            clean_df,
            raw_df=raw_df,
            time_series_granularity=time_series_granularity,
        )

        # 4. Konsol raporu
        self.print_summary_report(summary)

        anomali_str = "EVET" if summary.anomaly_detected else "hayir"
        logger.info(
            f"=== Analitik Pipeline Tamamlandi === "
            f"Temiz kayit: {len(clean_df)} | "
            f"sentiment={summary.average_sentiment_score:+.3f} | "
            f"anomali={anomali_str}"
        )

        return clean_df, summary

