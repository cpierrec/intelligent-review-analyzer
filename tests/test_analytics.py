"""
tests/test_analytics.py
========================
analytics.py için pytest unit testleri.

Test kapsamı:
  - reviews_to_dataframe(): Pydantic → DataFrame dönüşümü, review_date korunumu
  - clean_dataframe(): spam filtresi, düşük güven eleme, dedup, rating range
  - compute_time_series(): periyot aggregation, negative_rate hesabı
  - compute_anomaly_detection(): z-score eşik tespiti, az periyot edge case
  - compute_summary(): AnalyticsSummary KPI'ları, anomali entegrasyonu

Tasarım kararları:
  - @pytest.fixture ile merkezi DataFrame/review veri seti
  - Parametrize: granularity seçenekleri
  - Assertion'lar hem tip hem değer kontrolü yapar
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone

import pandas as pd

from analytics import DataAnalytics
from models import (
    AIAnalyzedReview,
    AIBatchResult,
    AnalyticsSummary,
    DataQuality,
    SentimentLabel,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_review(
    source_id: str,
    sentiment: str,
    score: float,
    confidence: float,
    rating: float | None = None,
    spam: bool = False,
    urgency: str = "low",
    review_date: datetime | None = None,
    topics: list[str] | None = None,
) -> AIAnalyzedReview:
    """AIAnalyzedReview factory yardımcı fonksiyonu."""
    return AIAnalyzedReview(
        source_id=source_id,
        product_name="Test Product Alpha",
        sentiment=sentiment,
        sentiment_score=score,
        confidence_score=confidence,
        summary="This is a test review summary that is long enough for validation.",
        reviewer_rating=rating,
        is_spam_or_fake=spam,
        urgency_level=urgency,
        review_date=review_date,
        key_topics=topics or [],
    )


@pytest.fixture
def sample_reviews() -> list[AIAnalyzedReview]:
    """5 çeşit review içeren temel test fixture'ı."""
    return [
        _make_review("r1", "positive", 0.85, 0.92, rating=5.0,
                     review_date=datetime(2024, 3, 10, tzinfo=timezone.utc),
                     topics=["battery", "sound"]),
        _make_review("r2", "negative", -0.70, 0.88, rating=2.0,
                     review_date=datetime(2024, 3, 10, tzinfo=timezone.utc),
                     topics=["sound", "price"]),
        _make_review("r3", "negative", -0.90, 0.95, rating=1.0, urgency="critical",
                     review_date=datetime(2024, 3, 11, tzinfo=timezone.utc)),
        _make_review("r4", "positive", 0.60, 0.78, rating=4.0,
                     review_date=datetime(2024, 3, 12, tzinfo=timezone.utc),
                     topics=["battery"]),
        _make_review("r5", "neutral", 0.00, 0.65,
                     review_date=datetime(2024, 3, 12, tzinfo=timezone.utc)),
    ]


@pytest.fixture
def spam_reviews() -> list[AIAnalyzedReview]:
    """Spam ve düşük güven içeren review'lar."""
    return [
        _make_review("s1", "positive", 0.9, 0.95, spam=True),
        _make_review("s2", "negative", -0.5, 0.30),  # Düşük güven
        _make_review("s3", "neutral", 0.0, 0.80),    # Temiz kayıt
    ]


@pytest.fixture
def analytics() -> DataAnalytics:
    """DataAnalytics instance."""
    return DataAnalytics()


# ---------------------------------------------------------------------------
# reviews_to_dataframe() Testleri
# ---------------------------------------------------------------------------

class TestReviewsToDataframe:
    """reviews_to_dataframe() metod testleri."""

    def test_empty_list_returns_empty_df(self, analytics: DataAnalytics) -> None:
        """Boş liste → boş DataFrame döndürmeli."""
        df = analytics.reviews_to_dataframe([])
        assert df.empty

    def test_correct_row_count(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """DataFrame satır sayısı review sayısına eşit olmalı."""
        df = analytics.reviews_to_dataframe(sample_reviews)
        assert len(df) == len(sample_reviews)

    def test_review_date_is_datetime(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """review_date sütunu datetime/Timestamp tipinde olmalı (string DEĞİL)."""
        df = analytics.reviews_to_dataframe(sample_reviews)
        assert "review_date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["review_date"])

    def test_key_topics_str_created(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """key_topics_str pipe-joined string sütunu oluşturulmalı."""
        df = analytics.reviews_to_dataframe(sample_reviews)
        assert "key_topics_str" in df.columns
        r1_topics = df[df["source_id"] == "r1"]["key_topics_str"].iloc[0]
        assert "battery" in r1_topics
        assert "|" in r1_topics  # Pipe separator

    def test_sentiment_normalized_lowercase(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """sentiment sütunu lowercase olmalı."""
        df = analytics.reviews_to_dataframe(sample_reviews)
        assert all(df["sentiment"].str.islower())


# ---------------------------------------------------------------------------
# clean_dataframe() Testleri
# ---------------------------------------------------------------------------

class TestCleanDataframe:
    """clean_dataframe() pipeline testleri."""

    def test_spam_removed(
        self, analytics: DataAnalytics, spam_reviews: list[AIAnalyzedReview]
    ) -> None:
        """Spam kayıtlar temizleme sonrası kaldırılmalı."""
        raw_df = analytics.reviews_to_dataframe(spam_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        assert "s1" not in clean_df["source_id"].values

    def test_low_confidence_removed(
        self, analytics: DataAnalytics, spam_reviews: list[AIAnalyzedReview]
    ) -> None:
        """Düşük güven skoru (<0.5) olan kayıtlar kaldırılmalı."""
        raw_df = analytics.reviews_to_dataframe(spam_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        assert "s2" not in clean_df["source_id"].values

    def test_clean_record_kept(
        self, analytics: DataAnalytics, spam_reviews: list[AIAnalyzedReview]
    ) -> None:
        """Temiz kayıt (spam yok, yüksek güven) korunmalı."""
        raw_df = analytics.reviews_to_dataframe(spam_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        assert "s3" in clean_df["source_id"].values

    def test_duplicate_source_id_removed(self, analytics: DataAnalytics) -> None:
        """Aynı source_id'ye sahip ikinci kayıt kaldırılmalı."""
        reviews = [
            _make_review("dup_1", "positive", 0.8, 0.9),
            _make_review("dup_1", "negative", -0.5, 0.85),  # Duplicate
        ]
        raw_df = analytics.reviews_to_dataframe(reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        assert len(clean_df[clean_df["source_id"] == "dup_1"]) == 1

    def test_sentiment_score_clipped(self, analytics: DataAnalytics) -> None:
        """sentiment_score -1.0/+1.0 aralığına clip edilmeli."""
        reviews = [_make_review("r_clip", "positive", 0.99, 0.9)]
        raw_df = analytics.reviews_to_dataframe(reviews)
        # Manüel olarak aşım değeri yaz
        raw_df.loc[0, "sentiment_score"] = 1.5
        clean_df = analytics.clean_dataframe(raw_df)
        assert clean_df.iloc[0]["sentiment_score"] <= 1.0

    def test_empty_df_returns_empty(self, analytics: DataAnalytics) -> None:
        """Boş DataFrame clean_dataframe'e verilirse boş döndürmeli."""
        result = analytics.clean_dataframe(pd.DataFrame())
        assert result.empty

    def test_index_reset_after_cleaning(
        self, analytics: DataAnalytics, spam_reviews: list[AIAnalyzedReview]
    ) -> None:
        """Temizleme sonrası index 0'dan başlamalı."""
        raw_df = analytics.reviews_to_dataframe(spam_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        assert list(clean_df.index) == list(range(len(clean_df)))


# ---------------------------------------------------------------------------
# compute_time_series() Testleri
# ---------------------------------------------------------------------------

class TestComputeTimeSeries:
    """compute_time_series() zaman serisi analiz testleri."""

    def test_correct_period_count(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """3 farklı tarih → 3 periyot döndürmeli."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        ts = analytics.compute_time_series(clean_df, granularity="day")
        assert len(ts) == 3

    def test_period_format_day(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """Günlük granülarite ile YYYY-MM-DD formatı beklenmeli."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        ts = analytics.compute_time_series(raw_df, granularity="day")
        for item in ts:
            assert len(item["period"]) == 10  # YYYY-MM-DD
            assert item["period"].count("-") == 2

    def test_negative_rate_calculation(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """2024-03-10'da 1 pozitif + 1 negatif → negative_rate=0.5."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        ts = analytics.compute_time_series(raw_df, granularity="day")
        day_10 = next(d for d in ts if d["period"] == "2024-03-10")
        assert day_10["negative_rate"] == pytest.approx(0.5)

    def test_100_percent_negative_rate(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """2024-03-11'de sadece 1 negatif kayıt → negative_rate=1.0."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        ts = analytics.compute_time_series(raw_df, granularity="day")
        day_11 = next(d for d in ts if d["period"] == "2024-03-11")
        assert day_11["negative_rate"] == pytest.approx(1.0)

    def test_no_review_date_returns_empty(self, analytics: DataAnalytics) -> None:
        """review_date olmayan DataFrame → boş liste döndürmeli."""
        df = pd.DataFrame({"sentiment_score": [0.5, -0.3], "source_id": ["a", "b"]})
        ts = analytics.compute_time_series(df, granularity="day")
        assert ts == []

    @pytest.mark.parametrize("granularity", ["day", "week"])
    def test_granularity_accepted(
        self,
        analytics: DataAnalytics,
        sample_reviews: list[AIAnalyzedReview],
        granularity: str,
    ) -> None:
        """Her iki granülarite de hata vermeden çalışmalı."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        ts = analytics.compute_time_series(raw_df, granularity=granularity)
        assert isinstance(ts, list)

    def test_time_series_dict_has_required_keys(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """Her periyot dict'i gerekli anahtarlara sahip olmalı."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        ts = analytics.compute_time_series(raw_df, granularity="day")
        required_keys = {"period", "avg_sentiment", "review_count", "negative_rate"}
        for item in ts:
            assert required_keys.issubset(set(item.keys()))


# ---------------------------------------------------------------------------
# compute_anomaly_detection() Testleri
# ---------------------------------------------------------------------------

class TestComputeAnomalyDetection:
    """compute_anomaly_detection() z-score anomali tespiti testleri."""

    def test_insufficient_periods_returns_no_anomaly(
        self, analytics: DataAnalytics
    ) -> None:
        """3'ten az periyot → anomali=False, detay=[] döndürmeli."""
        ts = [
            {"period": "2024-01-01", "negative_rate": 0.1, "avg_sentiment": 0.5},
            {"period": "2024-01-02", "negative_rate": 0.2, "avg_sentiment": 0.4},
        ]
        detected, details = analytics.compute_anomaly_detection(ts)
        assert detected is False
        assert details == []

    def test_anomaly_detected_with_spike(self, analytics: DataAnalytics) -> None:
        """Belirgin spike → anomali tespit edilmeli (z-score > 2)."""
        # 7 normal periyot + 1 anormal spike
        ts = [
            {"period": f"2024-01-0{i}", "negative_rate": 0.1, "avg_sentiment": 0.5}
            for i in range(1, 8)
        ]
        ts.append({"period": "2024-01-08", "negative_rate": 0.95, "avg_sentiment": -0.9})
        detected, details = analytics.compute_anomaly_detection(
            ts, metrics=["negative_rate"], z_threshold=2.0
        )
        assert detected is True
        assert len(details) >= 1

    def test_anomaly_detail_structure(self, analytics: DataAnalytics) -> None:
        """Anomali detay dict'i gerekli anahtarlara sahip olmalı."""
        ts = [
            {"period": f"2024-01-0{i}", "negative_rate": 0.1, "avg_sentiment": 0.5}
            for i in range(1, 8)
        ]
        ts.append({"period": "2024-01-08", "negative_rate": 0.99, "avg_sentiment": -0.95})
        _, details = analytics.compute_anomaly_detection(ts, metrics=["negative_rate"])
        if details:
            required_keys = {"period", "metric", "value", "z_score", "severity"}
            assert required_keys.issubset(set(details[0].keys()))

    def test_constant_values_no_anomaly(self, analytics: DataAnalytics) -> None:
        """Tüm değerler aynıysa z-score tanımsız → anomali yok."""
        ts = [
            {"period": f"2024-01-0{i}", "negative_rate": 0.5, "avg_sentiment": 0.0}
            for i in range(1, 8)
        ]
        detected, details = analytics.compute_anomaly_detection(ts)
        assert detected is False

    def test_severity_levels(self, analytics: DataAnalytics) -> None:
        """Farklı z-score büyüklükleri için doğru severity atanmalı."""
        # 99×0.0 + 1×100.0
        # mean=1.0, std(ddof=1)≈10.05, z=(100-1)/10.05≈9.85 → "critical" garantili
        ts = [{"period": f"d{i}", "negative_rate": 0.0} for i in range(99)]
        ts.append({"period": "d99", "negative_rate": 100.0})

        _, details = analytics.compute_anomaly_detection(
            ts, metrics=["negative_rate"], z_threshold=2.0
        )
        assert len(details) >= 1, "Büyük spike anomali üretmeli"
        severities = {d["severity"] for d in details}
        # z ≈ 9.85 >> 3.0 → "critical" kesin
        assert "critical" in severities


# ---------------------------------------------------------------------------
# compute_summary() Entegrasyon Testleri
# ---------------------------------------------------------------------------

class TestComputeSummary:
    """compute_summary() KPI hesaplama entegrasyon testleri."""

    def test_total_reviews_count(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """total_reviews_analyzed temizlenmiş kayıt sayısına eşit olmalı."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        summary = analytics.compute_summary(clean_df)
        assert summary.total_reviews_analyzed == len(clean_df)

    def test_sentiment_distribution_populated(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """sentiment_distribution boş olmamalı."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        summary = analytics.compute_summary(clean_df)
        assert len(summary.sentiment_distribution) > 0

    def test_time_series_populated_with_dates(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """review_date olan veri varsa time_series_data dolmalı."""
        raw_df = analytics.reviews_to_dataframe(sample_reviews)
        clean_df = analytics.clean_dataframe(raw_df)
        summary = analytics.compute_summary(clean_df)
        assert len(summary.time_series_data) > 0

    def test_empty_df_returns_empty_summary(self, analytics: DataAnalytics) -> None:
        """Boş DataFrame → report_id olan boş AnalyticsSummary döndürmeli."""
        summary = analytics.compute_summary(pd.DataFrame())
        assert isinstance(summary, AnalyticsSummary)
        assert summary.total_reviews_analyzed == 0

    def test_run_pipeline_full_integration(
        self, analytics: DataAnalytics, sample_reviews: list[AIAnalyzedReview]
    ) -> None:
        """run_pipeline() tam entegrasyon — clean_df ve summary döndürmeli."""
        batch = AIBatchResult(
            batch_id="test_batch",
            total_input=len(sample_reviews),
            successful=sample_reviews,
        )
        clean_df, summary = analytics.run_pipeline(batch)
        assert not clean_df.empty
        assert isinstance(summary, AnalyticsSummary)
        assert summary.total_reviews_analyzed > 0
