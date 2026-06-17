# AI-Powered Automation & Web Data Integration Engine

A production-grade Python pipeline that scrapes product review data, runs it through LLM APIs (OpenAI / Anthropic / Mock), extracts structured business intelligence, detects anomalies, and exports the results as Excel, CSV, and JSON — all from a single CLI command.

---

## What it actually does

You point it at a data source. It fetches reviews, sends them to an LLM in token-aware batches, validates every response against a strict Pydantic schema, cleans the resulting DataFrame, runs time-series KPI analysis with z-score anomaly detection, and writes a colour-coded Excel workbook to disk.

The whole thing runs in under 15 seconds on 20 records with the mock provider. Swap in a real API key and it scales.

---

## Architecture

```
Input (CSV / JSON / web)
         │
         ▼
┌─────────────────┐
│  Layer 1        │  scraper.py
│  Data Ingestion │  asyncio + httpx, User-Agent pool,
│                 │  ProxyRotator, jitter sleep,
│                 │  graceful fallback to requests
└────────┬────────┘
         │  list[RawReviewData]
         ▼
┌─────────────────┐
│  Layer 2        │  ai_processor.py
│  AI Processing  │  Strategy pattern: OpenAI / Anthropic / Mock
│                 │  tiktoken budget, Structured Outputs (json_schema),
│                 │  exponential backoff, async parallel batching
└────────┬────────┘
         │  AIBatchResult
         ▼
┌─────────────────┐
│  Layer 3        │  analytics.py
│  Analytics      │  pandas vectorized pipeline, spam filter,
│                 │  confidence threshold, dedup, time-series
│                 │  (day/week), z-score anomaly detection
└────────┬────────┘
         │  (clean_df, AnalyticsSummary)
         ▼
┌─────────────────┐
│  Layer 4        │  reporter.py
│  Export         │  JSON, CSV (UTF-8 BOM), Excel with
│                 │  colour-coded rows, KPI sheet, bar chart
└─────────────────┘
```

Everything is wired together in `main.py`, which exposes a CLI and handles errors at each layer boundary independently.

---

## Setup

```bash
git clone https://github.com/your-username/ai-data-engine.git
cd ai-data-engine

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

cp .env.example .env
# Open .env and set your keys — or leave AI_PROVIDER=mock to run without any API key
```

---

## Running

```bash
# Quickstart — no API key needed
python main.py

# Feed a local JSON file instead of scraping
python main.py --input data/reviews.json

# Weekly time-series, async batch processing, save to a custom path
python main.py --input data/reviews.csv --granularity week --async-mode --output reports/q2

# Real OpenAI, 50 records, skip Excel
python main.py --provider openai --records 50 --no-excel

# Full debug output
python main.py --provider anthropic --log-level DEBUG
```

After a successful run:

```
reports/
├── reviews_analysis_20240615_143022.json   ← full dataset + KPI summary
├── reviews_analysis_20240615_143022.csv    ← flat table, UTF-8 BOM
└── reviews_analysis_20240615_143022.xlsx   ← colour-coded workbook (3 sheets)
```

---

## CLI reference

```
python main.py [OPTIONS]

  --provider    {mock,openai,anthropic}     AI backend (default: mock)
  --input       FILE                        .json or .csv input file
  --source      {mock_api,html_scrape}      Data source when --input is not set
  --records     N                           Max records to process (default: 20)
  --output      PATH                        Output directory / file base
  --no-excel                                Skip .xlsx export
  --granularity {day,week}                  Time-series granularity (default: day)
  --async-mode                              Process batches in parallel (asyncio.gather)
  --max-concurrent N                        Semaphore limit for async mode (default: 3)
  --log-level   {DEBUG,INFO,WARNING,ERROR}  Log verbosity (default: INFO)
```

---

## Input file format

**JSON** (`list` of objects):
```json
[
  {"source_id": "r001", "raw_text": "Title: Great headphones\nReview: Crystal clear audio..."},
  {"source_id": "r002", "raw_text": "Title: Broken on arrival\nReview: Stopped working after 2 days."}
]
```

**CSV** (header row required):
```
source_id,raw_text
r001,"Title: Great headphones\nReview: Crystal clear audio..."
r002,"Title: Broken on arrival\nReview: Stopped working after 2 days."
```

Malformed rows are skipped with a warning — the rest of the pipeline continues.

---

## Key technical decisions

### Token-aware batch splitting

Before any API call, `TokenCounter` measures the real token count of each record using `tiktoken`. `BatchSplitter` then groups records into sub-batches that fit within the configured token budget. This prevents context-window overflows and keeps per-call costs predictable.

```python
# config default: 6 000 tokens per batch, 800 token schema overhead
batch_splitter = BatchSplitter(token_counter, max_tokens_per_batch=6000)
```

### OpenAI Structured Outputs

The `json_schema` response format with `"strict": true` is used instead of the older `json_object` mode. The schema is auto-generated from the `AIAnalyzedReview` Pydantic model and cached — so the prompt and the model definition can never drift out of sync.

### Z-score anomaly detection

After time-series aggregation, `compute_anomaly_detection()` calculates a z-score for each metric across all periods. Anything beyond the threshold triggers an alert with severity tiering:

| z-score | severity |
|---------|----------|
| > 3.0   | critical |
| > 2.5   | high     |
| > 2.0   | medium   |

A spike in `negative_rate` on a specific day will surface as a `CRITICAL` anomaly in the report.

### Async processing

`process_all_async()` splits records into sub-batches the same way the synchronous version does, but fires them all via `asyncio.gather()` with a configurable `Semaphore` to respect rate limits. The mock provider returns instantly; real API providers see wall-time reduction proportional to the number of concurrent batches.

### Scraper resilience

- **User-Agent pool**: 15 real browser signatures rotated per request
- **ProxyRotator**: round-robin selection; `mark_failed()` removes a proxy from rotation without stopping the scrape
- **Jitter sleep**: `base ± uniform(0, jitter)` delay between requests
- **Graceful degradation**: if `httpx` is not installed, falls back to `requests` transparently

---

## Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `AI_PROVIDER` | `mock` | `mock` / `openai` / `anthropic` |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `ANTHROPIC_API_KEY` | — | Required when provider is `anthropic` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any chat-completion model |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | |
| `AI_MAX_TOKENS` | `1500` | Max tokens per LLM response |
| `AI_BATCH_SIZE` | `5` | Records per API call |
| `AI_MAX_RETRIES` | `3` | Retry attempts on API failure |
| `DATA_SOURCE_TYPE` | `mock_api` | `mock_api` / `html_scrape` |
| `MAX_RECORDS` | `20` | Pipeline record cap |
| `NET_MAX_RETRIES` | `5` | HTTP retry attempts |
| `NET_BACKOFF_FACTOR` | `0.5` | Exponential backoff multiplier |
| `LOG_LEVEL` | `INFO` | |
| `REPORT_OUTPUT_DIR` | `reports` | |

---

## Extending the engine

### New AI provider

Subclass `BaseAIProvider` and implement three methods:

```python
class GoogleGeminiProvider(BaseAIProvider):
    @property
    def model_name(self) -> str:
        return "gemini-1.5-flash"

    def analyze_single(self, raw_text: str, source_id: str) -> dict[str, Any]:
        ...

    def analyze_batch(
        self, reviews: list[RawReviewData]
    ) -> list[AIAnalyzedReview]:
        ...
```

Then register it in `create_ai_provider()`:

```python
registry = {
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini":    GoogleGeminiProvider,   # ← add here
    "mock":      MockAIProvider,
}
```

### New data source

Add a branch in `DataScraper.fetch_all()`:

```python
elif self.data_cfg.source_type == "rss_feed":
    return self._fetch_from_rss(self.data_cfg.rss_url)
```

### New export format

Add a method to `DataReporter` and call it from `export_all()`:

```python
if "parquet" in self._config.export_formats:
    results["parquet"] = self._export_parquet(clean_df, base_path)
```

---

## Tests

```bash
python -m pytest tests/ -v
```

111 tests across three files:

| File | Coverage |
|------|----------|
| `tests/test_models.py` | Pydantic validators, business rules, edge cases |
| `tests/test_analytics.py` | DataFrame fixtures, cleaning pipeline, time-series, anomaly detection |
| `tests/test_ai_processor.py` | TokenCounter, BatchSplitter, MockAIProvider (EN+TR), integration |

No mocking of external APIs needed — the MockAIProvider is content-aware and deterministic by design.

---

## Project structure

```
ai_data_engine/
├── config.py          # Environment config, logging setup
├── models.py          # Pydantic models, JSON schema generation
├── scraper.py         # Async HTTP scraping, UA pool, proxy rotation
├── ai_processor.py    # LLM integration, batch splitting, retry
├── analytics.py       # pandas pipeline, time-series, anomaly detection
├── reporter.py        # JSON / CSV / Excel export
├── main.py            # CLI entrypoint, pipeline orchestration
├── tests/
│   ├── test_models.py
│   ├── test_analytics.py
│   └── test_ai_processor.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## License

MIT
