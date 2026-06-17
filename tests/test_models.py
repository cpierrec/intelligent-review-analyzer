"""
tests/test_models.py
====================
models.py için pytest unit testleri.

Test kapsamı:
  - RawReviewData: source_id/raw_text validasyonu, edge case'ler
  - AIAnalyzedReview: rating range, confidence normalizasyon, sentiment clip,
    business rules (@model_validator), spam auto-quality
  - AIBatchResult: success_rate property
  - AnalyticsSummary: default alanlar

Tasarım kararları:
  - Parametrize testler → aynı logic farklı girdilerle test
  - pytest.raises → Pydantic ValidationError yakala
  - fixture → tekrar eden model oluşturmayı merkezi hale getir
"""

from __future__ import annotations

import sys
import os

# ai_data_engine klasörünü Python path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from models import (
    AIAnalyzedReview,
    AIBatchResult,
    AnalyticsSummary,
    DataQuality,
    ProductCategory,
    RawReviewData,
    SentimentLabel,
    UrgencyLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_raw_review() -> dict:
    """Geçerli bir RawReviewData dict'i."""
    return {
        "source_id": "comment_42",
        "source_url": "https://example.com/product/1",
        "raw_text": "This product is absolutely amazing and exceeded all my expectations!",
        "metadata": {"star_rating": 5},
    }


@pytest.fixture
def valid_analyzed_review() -> dict:
    """Geçerli bir AIAnalyzedReview dict'i (minimum zorunlu alanlar)."""
    return {
        "source_id": "r_001",
        "product_name": "Wireless Headphones Pro",
        "sentiment": "positive",
        "sentiment_score": 0.85,
        "confidence_score": 0.92,
        "summary": "Customer is highly satisfied with sound quality and battery life.",
    }


# ---------------------------------------------------------------------------
# RawReviewData Testleri
# ---------------------------------------------------------------------------

class TestRawReviewData:
    """RawReviewData Pydantic modelinin validasyon testleri."""

    def test_valid_creation(self, valid_raw_review: dict) -> None:
        """Geçerli data ile model oluşturma başarılı olmalı."""
        model = RawReviewData(**valid_raw_review)
        assert model.source_id == "comment_42"
        assert len(model.raw_text) > 10

    def test_source_id_empty_string_raises(self) -> None:
        """Boş source_id ValidationError fırlatmalı."""
        with pytest.raises(ValidationError) as exc_info:
            RawReviewData(source_id="", raw_text="This is a valid review text here.")
        errors = exc_info.value.errors()
        assert any("source_id" in str(e) for e in errors)

    def test_source_id_whitespace_only_raises(self) -> None:
        """Sadece boşluktan oluşan source_id reddedilmeli."""
        with pytest.raises(ValidationError):
            RawReviewData(source_id="   ", raw_text="This is a valid review text here.")

    def test_source_id_stripped(self) -> None:
        """source_id etrafındaki boşluklar trim edilmeli."""
        model = RawReviewData(
            source_id="  comment_42  ",
            raw_text="This is a valid review text here.",
        )
        assert model.source_id == "comment_42"

    def test_raw_text_too_short_raises(self) -> None:
        """10 karakterden kısa raw_text ValidationError fırlatmalı."""
        with pytest.raises(ValidationError) as exc_info:
            RawReviewData(source_id="r1", raw_text="Too short")
        errors = exc_info.value.errors()
        assert any("raw_text" in str(e) for e in errors)

    def test_raw_text_exactly_10_chars_valid(self) -> None:
        """Tam 10 karakter geçerli olmalı."""
        model = RawReviewData(source_id="r1", raw_text="1234567890")
        assert len(model.raw_text) == 10

    def test_raw_text_stripped(self) -> None:
        """raw_text başı/sonu boşlukları trim edilmeli."""
        model = RawReviewData(
            source_id="r1",
            raw_text="   This is a review with leading spaces.   ",
        )
        assert not model.raw_text.startswith(" ")
        assert not model.raw_text.endswith(" ")

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """metadata belirtilmezse boş dict olmalı."""
        model = RawReviewData(source_id="r1", raw_text="Valid review text here ok.")
        assert model.metadata == {}

    def test_scraped_at_auto_set(self, valid_raw_review: dict) -> None:
        """scraped_at otomatik olarak şimdiki zaman olarak atanmalı."""
        model = RawReviewData(**valid_raw_review)
        assert model.scraped_at is not None
        assert model.scraped_at.tzinfo is not None  # timezone-aware


# ---------------------------------------------------------------------------
# AIAnalyzedReview - Field Validator Testleri
# ---------------------------------------------------------------------------

class TestAIAnalyzedReviewFieldValidators:
    """AIAnalyzedReview field-level validator testleri."""

    def test_valid_review_creation(self, valid_analyzed_review: dict) -> None:
        """Geçerli data ile model oluşturma başarılı olmalı."""
        review = AIAnalyzedReview(**valid_analyzed_review)
        assert review.source_id == "r_001"
        assert review.sentiment == SentimentLabel.POSITIVE

    # --- reviewer_rating ---

    @pytest.mark.parametrize("rating", [1.0, 2.5, 3.0, 4.5, 5.0])
    def test_valid_ratings_accepted(
        self, valid_analyzed_review: dict, rating: float
    ) -> None:
        """1.0-5.0 arası rating değerleri kabul edilmeli."""
        review = AIAnalyzedReview(**{**valid_analyzed_review, "reviewer_rating": rating})
        assert review.reviewer_rating == rating

    @pytest.mark.parametrize("invalid_rating", [0.0, 0.5, 5.1, 6.0, -1.0])
    def test_invalid_ratings_become_none(
        self, valid_analyzed_review: dict, invalid_rating: float
    ) -> None:
        """Geçersiz rating değerleri None'a dönüştürülmeli (exception yok)."""
        review = AIAnalyzedReview(
            **{**valid_analyzed_review, "reviewer_rating": invalid_rating}
        )
        assert review.reviewer_rating is None

    def test_none_rating_accepted(self, valid_analyzed_review: dict) -> None:
        """reviewer_rating=None geçerli olmalı."""
        review = AIAnalyzedReview(**{**valid_analyzed_review, "reviewer_rating": None})
        assert review.reviewer_rating is None

    # --- confidence_score ---

    @pytest.mark.parametrize("conf", [0.0, 0.5, 0.99, 1.0])
    def test_valid_confidence_accepted(
        self, valid_analyzed_review: dict, conf: float
    ) -> None:
        """0.0-1.0 arası confidence değerleri kabul edilmeli."""
        review = AIAnalyzedReview(**{**valid_analyzed_review, "confidence_score": conf})
        assert 0.0 <= review.confidence_score <= 1.0

    def test_confidence_100_scale_normalized(self, valid_analyzed_review: dict) -> None:
        """0-100 skala confidence değeri 0-1 aralığına normalize edilmeli."""
        review = AIAnalyzedReview(**{**valid_analyzed_review, "confidence_score": 92.0})
        assert abs(review.confidence_score - 0.92) < 0.001

    def test_confidence_out_of_range_raises(self, valid_analyzed_review: dict) -> None:
        """0-100 dışındaki confidence değeri ValidationError fırlatmalı."""
        with pytest.raises(ValidationError):
            AIAnalyzedReview(**{**valid_analyzed_review, "confidence_score": 150.0})

    def test_confidence_none_raises(self, valid_analyzed_review: dict) -> None:
        """confidence_score=None ValidationError fırlatmalı."""
        with pytest.raises(ValidationError):
            AIAnalyzedReview(**{**valid_analyzed_review, "confidence_score": None})

    # --- sentiment_score ---

    @pytest.mark.parametrize("score", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_valid_sentiment_scores(
        self, valid_analyzed_review: dict, score: float
    ) -> None:
        """Geçerli sentiment_score değerleri korunmalı."""
        review = AIAnalyzedReview(**{**valid_analyzed_review, "sentiment_score": score})
        assert review.sentiment_score == score

    @pytest.mark.parametrize("overflow_score,expected", [(-2.0, -1.0), (1.5, 1.0)])
    def test_sentiment_score_clipped(
        self,
        valid_analyzed_review: dict,
        overflow_score: float,
        expected: float,
    ) -> None:
        """Aralık dışı sentiment_score değerleri clip edilmeli."""
        review = AIAnalyzedReview(
            **{**valid_analyzed_review, "sentiment_score": overflow_score}
        )
        assert review.sentiment_score == expected

    # --- reviewer_expertise ---

    @pytest.mark.parametrize("expertise", ["novice", "intermediate", "expert"])
    def test_valid_expertise_values(
        self, valid_analyzed_review: dict, expertise: str
    ) -> None:
        """İzin verilen uzmanlık değerleri kabul edilmeli."""
        review = AIAnalyzedReview(
            **{**valid_analyzed_review, "reviewer_expertise": expertise}
        )
        assert review.reviewer_expertise == expertise

    def test_invalid_expertise_becomes_none(self, valid_analyzed_review: dict) -> None:
        """Bilinmeyen uzmanlık değeri None'a dönüştürülmeli."""
        review = AIAnalyzedReview(
            **{**valid_analyzed_review, "reviewer_expertise": "wizard"}
        )
        assert review.reviewer_expertise is None

    # --- key_topics / pros / cons (list coercion) ---

    def test_comma_separated_string_coerced_to_list(
        self, valid_analyzed_review: dict
    ) -> None:
        """Virgülle ayrılmış string liste'ye çevrilmeli."""
        review = AIAnalyzedReview(
            **{**valid_analyzed_review, "key_topics": "battery, sound, design"}
        )
        assert isinstance(review.key_topics, list)
        assert len(review.key_topics) == 3
        assert "battery" in review.key_topics

    def test_none_list_field_becomes_empty_list(
        self, valid_analyzed_review: dict
    ) -> None:
        """None list alanları boş liste olmalı."""
        review = AIAnalyzedReview(**{**valid_analyzed_review, "key_topics": None})
        assert review.key_topics == []


# ---------------------------------------------------------------------------
# AIAnalyzedReview - Business Rules (@model_validator) Testleri
# ---------------------------------------------------------------------------

class TestAIAnalyzedReviewBusinessRules:
    """Cross-field iş kuralları testleri."""

    def test_negative_low_rating_upgrades_urgency(
        self, valid_analyzed_review: dict
    ) -> None:
        """NEGATIVE sentiment + rating<=2.0 → urgency=HIGH, action_required=True."""
        review = AIAnalyzedReview(**{
            **valid_analyzed_review,
            "sentiment": "negative",
            "sentiment_score": -0.8,
            "reviewer_rating": 1.5,
            "urgency_level": "low",
        })
        assert review.urgency_level == UrgencyLevel.HIGH
        assert review.action_required is True

    def test_negative_high_rating_no_urgency_change(
        self, valid_analyzed_review: dict
    ) -> None:
        """NEGATIVE sentiment ama rating>2.0 → urgency değişmemeli."""
        review = AIAnalyzedReview(**{
            **valid_analyzed_review,
            "sentiment": "negative",
            "sentiment_score": -0.3,
            "reviewer_rating": 3.0,
            "urgency_level": "low",
        })
        # Kural 1: rating>2.0 olduğu için tetiklenmiyor
        assert review.urgency_level == UrgencyLevel.LOW

    def test_critical_urgency_forces_action_required(
        self, valid_analyzed_review: dict
    ) -> None:
        """CRITICAL urgency → action_required her zaman True."""
        review = AIAnalyzedReview(**{
            **valid_analyzed_review,
            "urgency_level": "critical",
            "action_required": False,  # Override edilmeli
        })
        assert review.action_required is True

    def test_spam_sets_data_quality_low(self, valid_analyzed_review: dict) -> None:
        """is_spam_or_fake=True → data_quality otomatik LOW."""
        review = AIAnalyzedReview(**{
            **valid_analyzed_review,
            "is_spam_or_fake": True,
            "data_quality": "high",  # Override edilmeli
        })
        assert review.data_quality == DataQuality.LOW

    def test_positive_sentiment_no_urgency_change(
        self, valid_analyzed_review: dict
    ) -> None:
        """Pozitif sentiment business rules'ı tetiklememeli."""
        review = AIAnalyzedReview(**valid_analyzed_review)
        assert review.urgency_level == UrgencyLevel.LOW
        assert review.action_required is False


# ---------------------------------------------------------------------------
# AIBatchResult Testleri
# ---------------------------------------------------------------------------

class TestAIBatchResult:
    """AIBatchResult model ve property testleri."""

    def test_success_rate_calculation(self, valid_analyzed_review: dict) -> None:
        """success_rate = successful / total_input."""
        r1 = AIAnalyzedReview(**valid_analyzed_review)
        r2 = AIAnalyzedReview(**{**valid_analyzed_review, "source_id": "r_002"})
        batch = AIBatchResult(
            batch_id="b001",
            total_input=5,
            successful=[r1, r2],
        )
        assert batch.success_rate == pytest.approx(0.4)

    def test_success_rate_zero_input(self) -> None:
        """total_input=0 → success_rate=0.0 (ZeroDivisionError yok)."""
        batch = AIBatchResult(batch_id="b002", total_input=0)
        assert batch.success_rate == 0.0

    def test_failed_count_property(self, valid_analyzed_review: dict) -> None:
        """failed_count = len(failed_ids)."""
        batch = AIBatchResult(
            batch_id="b003",
            total_input=3,
            failed_ids=["r1", "r2"],
        )
        assert batch.failed_count == 2

    def test_token_tracking_fields(self) -> None:
        """total_tokens_used ve estimated_cost_usd alanları default 0 olmalı."""
        batch = AIBatchResult(batch_id="b004", total_input=10)
        assert batch.total_tokens_used == 0
        assert batch.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# AnalyticsSummary Testleri
# ---------------------------------------------------------------------------

class TestAnalyticsSummary:
    """AnalyticsSummary model testleri."""

    def test_default_values(self) -> None:
        """Minimum alanlarla oluşturma — tüm default'lar doğru olmalı."""
        summary = AnalyticsSummary(report_id="test-001")
        assert summary.total_reviews_analyzed == 0
        assert summary.anomaly_detected is False
        assert summary.time_series_data == []
        assert summary.anomaly_details == []

    def test_generated_at_is_timezone_aware(self) -> None:
        """generated_at timezone-aware datetime olmalı."""
        summary = AnalyticsSummary(report_id="test-002")
        assert summary.generated_at.tzinfo is not None

    def test_schema_cache_returns_string(self) -> None:
        """get_llm_schema() JSON string döndürmeli ve cache çalışmalı."""
        schema1 = AIAnalyzedReview.get_llm_schema()
        schema2 = AIAnalyzedReview.get_llm_schema()
        assert isinstance(schema1, str)
        assert schema1 == schema2  # Cache'den geldiği için aynı nesne
        assert "sentiment_score" in schema1
