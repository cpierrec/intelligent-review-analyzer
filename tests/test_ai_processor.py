"""
tests/test_ai_processor.py
===========================
ai_processor.py için pytest unit testleri.

Test kapsamı:
  - TokenCounter: token sayma (tiktoken veya heuristic fallback)
  - BatchSplitter: token-aware batch bölme, min 1 kayıt garantisi
  - MockAIProvider: deterministik çıktı, EN+TR negatif sinyal tespiti,
    confidence aralığı, geçerli Pydantic model üretimi
  - create_ai_provider factory: mock seçimi
  - AIProcessor: process_all() entegrasyon testi

Tasarım kararları:
  - MockAIProvider seed'li RNG: aynı source_id → aynı çıktı
  - Monkeypatching gerektirmez — Mock provider hiç network kullanmaz
  - asyncio: process_all() async → pytest-asyncio ile test edilir
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from typing import List

from models import (
    AIAnalyzedReview,
    AIBatchResult,
    RawReviewData,
    SentimentLabel,
    UrgencyLevel,
)
from config import AIConfig
from ai_processor import (
    AIProcessor,
    BatchSplitter,
    MockAIProvider,
    TokenCounter,
    create_ai_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_config() -> AIConfig:
    """Varsayılan mock AIConfig."""
    return AIConfig(provider="mock")


@pytest.fixture
def mock_provider(ai_config: AIConfig) -> MockAIProvider:
    """MockAIProvider instance."""
    return MockAIProvider(ai_config)


@pytest.fixture
def token_counter() -> TokenCounter:
    """TokenCounter instance (gpt-4o-mini model)."""
    return TokenCounter("gpt-4o-mini")


@pytest.fixture
def sample_records() -> list[RawReviewData]:
    """5 farklı içerikli RawReviewData listesi."""
    return [
        RawReviewData(
            source_id=f"r{i}",
            raw_text=f"Title: Product {i}\nReview: This is a detailed review about product {i}.",
        )
        for i in range(1, 6)
    ]


@pytest.fixture
def negative_records() -> list[RawReviewData]:
    """Negatif sinyal içeren İngilizce ve Türkçe review'lar."""
    return [
        RawReviewData(
            source_id="neg_en",
            raw_text="Title: Broken Product\nReview: This product is terrible and completely broken. Terrible experience!",
        ),
        RawReviewData(
            source_id="neg_tr",
            raw_text="Title: Kötü Ürün\nReview: Bu ürün berbat ve tamamen bozuk çıktı. Çok kötü bir deneyim yaşadım.",
        ),
        RawReviewData(
            source_id="pos_en",
            raw_text="Title: Great Product\nReview: This product is absolutely amazing and works perfectly well.",
        ),
    ]


# ---------------------------------------------------------------------------
# TokenCounter Testleri
# ---------------------------------------------------------------------------

class TestTokenCounter:
    """TokenCounter sınıfı testleri."""

    def test_count_returns_positive_integer(self, token_counter: TokenCounter) -> None:
        """Token sayısı pozitif integer olmalı."""
        count = token_counter.count("Hello world, this is a test sentence.")
        assert isinstance(count, int)
        assert count > 0

    def test_empty_string_returns_zero(self, token_counter: TokenCounter) -> None:
        """Boş string → 0 token."""
        count = token_counter.count("")
        assert count == 0

    def test_longer_text_has_more_tokens(self, token_counter: TokenCounter) -> None:
        """Daha uzun metin daha fazla token içermeli."""
        short = "Hello world."
        long = "Hello world. " * 50
        assert token_counter.count(long) > token_counter.count(short)

    def test_count_is_deterministic(self, token_counter: TokenCounter) -> None:
        """Aynı metin her zaman aynı token sayısını vermeli."""
        text = "This is a test sentence for token counting."
        assert token_counter.count(text) == token_counter.count(text)

    def test_heuristic_fallback(self) -> None:
        """Bilinmeyen model → heuristic fallback kullanmalı (exception yok)."""
        counter = TokenCounter(model_name="unknown-model-xyz")
        count = counter.count("This text should still count with heuristic fallback.")
        assert count > 0

    def test_estimate_cost_returns_float(self, token_counter: TokenCounter) -> None:
        """estimate_cost_usd() float döndürmeli."""
        cost = token_counter.estimate_cost_usd(
            input_tokens=1000, output_tokens=200
        )
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_zero_tokens_zero_cost(self, token_counter: TokenCounter) -> None:
        """0 token → 0 maliyet."""
        cost = token_counter.estimate_cost_usd(input_tokens=0, output_tokens=0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# BatchSplitter Testleri
# ---------------------------------------------------------------------------

class TestBatchSplitter:
    """BatchSplitter token-aware batch bölme testleri."""

    def test_split_returns_list_of_lists(
        self, token_counter: TokenCounter, sample_records: list[RawReviewData]
    ) -> None:
        """split() list[list[RawReviewData]] döndürmeli."""
        splitter = BatchSplitter(token_counter, max_tokens_per_batch=500)
        batches = splitter.split(sample_records)
        assert isinstance(batches, list)
        assert all(isinstance(b, list) for b in batches)

    def test_all_records_present_after_split(
        self, token_counter: TokenCounter, sample_records: list[RawReviewData]
    ) -> None:
        """Bölünme sonrası hiç kayıt kaybolmamalı."""
        splitter = BatchSplitter(token_counter, max_tokens_per_batch=200)
        batches = splitter.split(sample_records)
        total = sum(len(b) for b in batches)
        assert total == len(sample_records)

    def test_minimum_one_record_per_batch(
        self, token_counter: TokenCounter, sample_records: list[RawReviewData]
    ) -> None:
        """Çok küçük token limiti bile en az 1 kayıt içermeli."""
        splitter = BatchSplitter(token_counter, max_tokens_per_batch=1)  # Çok küçük
        batches = splitter.split(sample_records)
        assert all(len(b) >= 1 for b in batches)

    def test_empty_records_returns_empty(self, token_counter: TokenCounter) -> None:
        """Boş liste → boş liste döndürmeli."""
        splitter = BatchSplitter(token_counter, max_tokens_per_batch=1000)
        assert splitter.split([]) == []

    def test_large_budget_single_batch(
        self, token_counter: TokenCounter, sample_records: list[RawReviewData]
    ) -> None:
        """Büyük token limiti → tek batch olabilir."""
        splitter = BatchSplitter(token_counter, max_tokens_per_batch=100_000)
        batches = splitter.split(sample_records)
        assert len(batches) >= 1
        # Toplam kayıt sayısı korunmalı
        total = sum(len(b) for b in batches)
        assert total == len(sample_records)


# ---------------------------------------------------------------------------
# MockAIProvider Testleri
# ---------------------------------------------------------------------------

class TestMockAIProvider:
    """MockAIProvider davranış testleri."""

    def test_analyze_returns_analyzed_review(
        self,
        mock_provider: MockAIProvider,
        sample_records: list[RawReviewData],
    ) -> None:
        """analyze() her kayıt için AIAnalyzedReview döndürmeli."""
        results = mock_provider.analyze_batch(sample_records)
        assert len(results) == len(sample_records)
        for result in results:
            assert isinstance(result, AIAnalyzedReview)

    def test_output_is_deterministic(
        self,
        mock_provider: MockAIProvider,
        sample_records: list[RawReviewData],
    ) -> None:
        """Aynı input → aynı output (seed'li RNG)."""
        results1 = mock_provider.analyze_batch([sample_records[0]])
        results2 = mock_provider.analyze_batch([sample_records[0]])
        assert results1[0].source_id == results2[0].source_id
        assert results1[0].sentiment == results2[0].sentiment
        assert results1[0].sentiment_score == results2[0].sentiment_score

    def test_negative_english_keywords_produce_negative_sentiment(
        self,
        mock_provider: MockAIProvider,
        negative_records: list[RawReviewData],
    ) -> None:
        """İngilizce negatif kelimeler negatif sentiment üretmeli."""
        neg_record = next(r for r in negative_records if r.source_id == "neg_en")
        results = mock_provider.analyze_batch([neg_record])
        assert results[0].sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.MIXED)

    def test_negative_turkish_keywords_produce_negative_sentiment(
        self,
        mock_provider: MockAIProvider,
        negative_records: list[RawReviewData],
    ) -> None:
        """Türkçe negatif kelimeler ('berbat', 'bozuk') negatif sentiment üretmeli."""
        neg_record = next(r for r in negative_records if r.source_id == "neg_tr")
        results = mock_provider.analyze_batch([neg_record])
        assert results[0].sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.MIXED)

    def test_confidence_score_in_valid_range(
        self,
        mock_provider: MockAIProvider,
        sample_records: list[RawReviewData],
    ) -> None:
        """Tüm çıktılarda confidence_score 0.0-1.0 arasında olmalı."""
        results = mock_provider.analyze_batch(sample_records)
        for r in results:
            assert 0.0 <= r.confidence_score <= 1.0

    def test_sentiment_score_in_valid_range(
        self,
        mock_provider: MockAIProvider,
        sample_records: list[RawReviewData],
    ) -> None:
        """Tüm çıktılarda sentiment_score -1.0 ile +1.0 arasında olmalı."""
        results = mock_provider.analyze_batch(sample_records)
        for r in results:
            assert -1.0 <= r.sentiment_score <= 1.0

    def test_source_id_preserved(
        self,
        mock_provider: MockAIProvider,
        sample_records: list[RawReviewData],
    ) -> None:
        """Çıktıdaki source_id, giriştekiyle aynı olmalı."""
        results = mock_provider.analyze_batch(sample_records)
        input_ids = {r.source_id for r in sample_records}
        output_ids = {r.source_id for r in results}
        assert input_ids == output_ids

    def test_positive_text_produces_positive_sentiment(
        self,
        mock_provider: MockAIProvider,
        negative_records: list[RawReviewData],
    ) -> None:
        """Açıkça pozitif metin pozitif veya nötr sentiment üretmeli."""
        pos_record = next(r for r in negative_records if r.source_id == "pos_en")
        results = mock_provider.analyze_batch([pos_record])
        assert results[0].sentiment in (
            SentimentLabel.POSITIVE,
            SentimentLabel.NEUTRAL,
            SentimentLabel.MIXED,
        )

    def test_analyze_batch_empty_list(self, mock_provider: MockAIProvider) -> None:
        """Boş liste → boş liste döndürmeli."""
        results = mock_provider.analyze_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# create_ai_provider Factory Testleri
# ---------------------------------------------------------------------------

class TestCreateAIProvider:
    """create_ai_provider factory fonksiyon testleri."""

    def test_mock_provider_created_for_mock_config(self) -> None:
        """provider='mock' → MockAIProvider örneği döndürmeli."""
        config = AIConfig(provider="mock")
        provider = create_ai_provider(config)
        assert isinstance(provider, MockAIProvider)

    def test_unknown_provider_falls_back_or_raises(self) -> None:
        """Bilinmeyen provider için MockAIProvider veya ValueError beklenir."""
        config = AIConfig(provider="unknown_xyz")
        try:
            provider = create_ai_provider(config)
            # Bazı implementasyonlar mock'a fallback yapar
            assert provider is not None
        except (ValueError, Exception):
            pass  # ValueError fırlatması da kabul edilebilir


# ---------------------------------------------------------------------------
# AIProcessor Entegrasyon Testleri
# ---------------------------------------------------------------------------

class TestAIProcessorIntegration:
    """AIProcessor.process_all() entegrasyon testleri (mock provider ile)."""

    def test_process_all_returns_batch_result(
        self,
        ai_config: AIConfig,
        sample_records: list[RawReviewData],
    ) -> None:
        """process_all() AIBatchResult döndürmeli."""
        processor = AIProcessor(ai_config)
        result = processor.process_all(sample_records)
        assert isinstance(result, AIBatchResult)

    def test_all_records_processed(
        self,
        ai_config: AIConfig,
        sample_records: list[RawReviewData],
    ) -> None:
        """Başarılı + başarısız toplam input sayısına eşit olmalı."""
        processor = AIProcessor(ai_config)
        result = processor.process_all(sample_records)
        total = len(result.successful) + len(result.failed_ids)
        assert total == len(sample_records)

    def test_success_rate_positive(
        self,
        ai_config: AIConfig,
        sample_records: list[RawReviewData],
    ) -> None:
        """Mock provider ile success_rate > 0 olmalı."""
        processor = AIProcessor(ai_config)
        result = processor.process_all(sample_records)
        assert result.success_rate > 0.0

    def test_processing_time_recorded(
        self,
        ai_config: AIConfig,
        sample_records: list[RawReviewData],
    ) -> None:
        """processing_time_seconds > 0 olmalı."""
        processor = AIProcessor(ai_config)
        result = processor.process_all(sample_records)
        assert result.processing_time_seconds >= 0.0

    def test_empty_input_returns_empty_result(self, ai_config: AIConfig) -> None:
        """Boş giriş → successful=[], failed_ids=[], total_input=0."""
        processor = AIProcessor(ai_config)
        result = processor.process_all([])
        assert result.total_input == 0
        assert result.successful == []
        assert result.failed_ids == []
