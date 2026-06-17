"""
main.py - Ana Orkestrasyon Modülü
===================================
Amaç: Tüm katmanları (Scraper → AI → Analytics → Reporter) sırayla
      çağıran ana giriş noktası. Pipeline mimarisini uygular.

Çalıştırma:
  python main.py                                    # Mock provider, mock kaynak
  python main.py --provider openai                  # Gerçek OpenAI API
  python main.py --provider anthropic               # Anthropic Claude
  python main.py --input data/reviews.json          # JSON/CSV dosyasından oku
  python main.py --output reports/my_report         # Özel çıktı yolu
  python main.py --granularity week                 # Haftalık zaman serisi
  python main.py --async-mode                       # Paralel sub-batch işleme
  python main.py --records 50 --log-level DEBUG     # 50 kayıt, detaylı log

Pipeline Akışı:
  [Config] → [Scraper/Input] → [AI Processor] → [Analytics] → [Reporter]
      ↓             ↓                  ↓               ↓             ↓
  AppConfig   RawReview[]        AIBatchResult    clean_df +     JSON/CSV/
              list               (structured)     Summary        XLSX files
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from config import AppConfig, setup_logging
from scraper import DataScraper, ScraperError
from ai_processor import AIProcessor, AIProcessorError
from analytics import DataAnalytics
from reporter import DataReporter
from models import RawReviewData


# ---------------------------------------------------------------------------
# CLI Argüman Ayrıştırıcı
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Komut satırı arayüzü tanımlar.

    12-Factor App prensibine göre ortam değişkenleri CLI argümanlarından
    önce gelir; CLI argümanları yalnızca geçici override için kullanılır.
    """
    parser = argparse.ArgumentParser(
        prog="ai-data-engine",
        description=(
            "AI-Powered Automation & Web Data Integration Engine\n"
            "Portfolio Project | Python Backend + AI Integration\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python main.py\n"
            "  python main.py --provider openai --records 30\n"
            "  python main.py --input data/reviews.json --output reports/out\n"
            "  python main.py --granularity week --async-mode\n"
            "  python main.py --provider anthropic --no-excel\n"
        ),
    )

    # ── AI Provider ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--provider",
        choices=["mock", "openai", "anthropic"],
        default=None,
        metavar="PROVIDER",
        help="AI sağlayıcı: mock | openai | anthropic (varsayılan: .env AI_PROVIDER veya 'mock')",
    )

    # ── Veri Kaynağı ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Girdi dosyası yolu (.json veya .csv). "
            "Belirtilmezse --source ile web kazıma veya mock API kullanılır."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["mock_api", "html_scrape"],
        default=None,
        help="Veri kaynağı tipi (--input verilmezse geçerli; varsayılan: 'mock_api')",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=None,
        metavar="N",
        help="İşlenecek maksimum kayıt sayısı (varsayılan: 20)",
    )

    # ── Çıktı ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Analiz sonuçları için çıktı dizini veya dosya tabanı "
            "(örn. reports/my_analysis). Uzantı otomatik eklenir."
        ),
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Excel (.xlsx) export'u atla",
    )

    # ── Analitik ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--granularity",
        choices=["day", "week"],
        default="day",
        help="Zaman serisi analizi granülaritesi: day | week (varsayılan: day)",
    )

    # ── Asenkron Mod ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help=(
            "Sub-batch'leri asyncio.gather ile paralel işle. "
            "Büyük veri setlerinde daha hızlı; ağ I/O bekleme sürelerini örtüşür."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        metavar="N",
        help="Async modda eşzamanlı sub-batch limiti (varsayılan: 3)",
    )

    # ── Genel ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log seviyesi (varsayılan: INFO)",
    )
    # Geriye dönük uyumluluk (eski --output-dir argümanı)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Gizli — --output tercih edilmeli
    )

    return parser


# ---------------------------------------------------------------------------
# Dosya Tabanlı Girdi Yükleyici
# ---------------------------------------------------------------------------

def load_records_from_file(file_path: str) -> list[RawReviewData]:
    """
    JSON veya CSV dosyasından RawReviewData listesi yükler.

    JSON formatı (list of objects):
      [{"source_id": "r1", "raw_text": "Review text..."}, ...]

    CSV formatı (başlık satırı zorunlu):
      source_id,raw_text
      r1,"Review text..."

    Hatalı satırlar atlanır; pipeline devam eder (graceful degradation).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {file_path}")

    records: list[RawReviewData] = []
    ext = path.suffix.lower()

    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"JSON dosyası list formatında olmalı, alınan: {type(data).__name__}"
            )
        for i, item in enumerate(data):
            try:
                records.append(RawReviewData(**item))
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"JSON satır #{i} parse hatası (atlanıyor): {e}"
                )

    elif ext == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    records.append(RawReviewData(**row))
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        f"CSV satır #{i + 2} parse hatası (atlanıyor): {e}"
                    )
    else:
        raise ValueError(
            f"Desteklenmeyen dosya formatı: '{ext}'. "
            "Desteklenenler: .json, .csv"
        )

    logging.getLogger(__name__).info(
        f"Dosyadan {len(records)} kayıt yüklendi: {file_path}"
    )
    return records


# ---------------------------------------------------------------------------
# Pipeline Orkestratörü
# ---------------------------------------------------------------------------

class DataPipelineOrchestrator:
    """
    Tüm katmanları koordine eden orkestratör sınıf.

    Her katman bağımsız hata yakalama ile korunur; bir katman başarısız
    olursa pipeline durur ve anlamlı hata mesajı verir.
    Bu yaklaşım production ortamında debugging süresini önemli ölçüde kısaltır.
    """

    def __init__(
        self,
        config: AppConfig,
        input_file: Optional[str] = None,
        granularity: str = "day",
        async_mode: bool = False,
        max_concurrent: int = 3,
    ) -> None:
        self._config = config
        self._input_file = input_file
        self._granularity = granularity
        self._async_mode = async_mode
        self._max_concurrent = max_concurrent
        self._logger = logging.getLogger(self.__class__.__name__)

    def _step_scrape(self) -> list[RawReviewData]:
        """
        Katman 1: Dosya okuma veya web kazıma.

        --input verilmişse dosyadan okur (JSON/CSV).
        Verilmemişse DataScraper ile web/mock API'den çeker.
        Çıktı: list[RawReviewData]
        """
        self._logger.info("▶ KATMAN 1/4: Veri Kazıma / Yükleme Başlıyor...")

        # Dosyadan okuma (--input argümanı)
        if self._input_file:
            try:
                records = load_records_from_file(self._input_file)
                if not records:
                    raise ScraperError(
                        f"Girdi dosyası boş veya parse edilemedi: {self._input_file}"
                    )
                # max_records limiti uygula
                if self._config.data_source.max_records:
                    records = records[: self._config.data_source.max_records]
                self._logger.info(
                    f"✓ Katman 1 tamamlandı: {len(records)} kayıt dosyadan yüklendi."
                )
                return records
            except (FileNotFoundError, ValueError) as e:
                self._logger.error(f"✗ Katman 1 HATA (dosya): {e}")
                raise ScraperError(str(e)) from e

        # Web/Mock kazıma
        try:
            scraper = DataScraper(
                network_cfg=self._config.network,
                data_cfg=self._config.data_source,
            )
            raw_records = scraper.fetch_all()

            if not raw_records:
                raise ScraperError(
                    "Scraper sıfır kayıt döndürdü. "
                    "Ağ bağlantısını ve kaynak URL'ini kontrol edin."
                )

            self._logger.info(
                f"✓ Katman 1 tamamlandı: {len(raw_records)} ham kayıt kazındı."
            )
            return raw_records

        except ScraperError:
            raise
        except Exception as e:
            self._logger.error(
                f"✗ Katman 1 beklenmeyen hata: {e}\n{traceback.format_exc()}"
            )
            raise ScraperError(f"Scraper beklenmeyen hata: {e}") from e

    def _step_ai_process(
        self, raw_records: list[RawReviewData]
    ):
        """
        Katman 2: AI entegrasyonu ve yapılandırılmış çıktı üretimi.

        --async-mode True ise process_all_async() çalışır (paralel sub-batch'ler).
        False ise process_all() çalışır (sıralı, öngörülebilir).
        Çıktı: AIBatchResult
        """
        self._logger.info(
            f"▶ KATMAN 2/4: AI Analizi Başlıyor... "
            f"({len(raw_records)} kayıt | provider={self._config.ai.provider.upper()} | "
            f"mod={'async' if self._async_mode else 'sync'})"
        )
        try:
            processor = AIProcessor(config=self._config.ai)

            if self._async_mode:
                # asyncio.run() → yeni event loop başlatır
                batch_result = asyncio.run(
                    processor.process_all_async(
                        raw_records,
                        max_concurrent=self._max_concurrent,
                    )
                )
            else:
                batch_result = processor.process_all(raw_records)

            if not batch_result.successful:
                raise AIProcessorError(
                    "AI hiçbir kaydı başarıyla işleyemedi. "
                    "API anahtarını ve model erişimini kontrol edin."
                )

            self._logger.info(
                f"✓ Katman 2 tamamlandı: "
                f"{len(batch_result.successful)}/{batch_result.total_input} kayıt analiz edildi "
                f"(başarı oranı: {batch_result.success_rate:.1%})"
            )
            return batch_result

        except AIProcessorError:
            raise
        except Exception as e:
            self._logger.error(
                f"✗ Katman 2 beklenmeyen hata: {e}\n{traceback.format_exc()}"
            )
            raise AIProcessorError(f"AI processor beklenmeyen hata: {e}") from e

    def _step_analytics(self, batch_result):
        """
        Katman 3: pandas veri analitiği ve KPI hesaplama.

        --granularity parametresi time-series granülaritesini belirler.
        Çıktı: (clean_df, AnalyticsSummary)
        """
        self._logger.info(
            f"▶ KATMAN 3/4: Veri Analitiği Başlıyor... "
            f"({len(batch_result.successful)} kayıt | granularity={self._granularity})"
        )
        try:
            analytics = DataAnalytics()
            clean_df, summary = analytics.run_pipeline(
                batch_result,
                time_series_granularity=self._granularity,
            )

            if clean_df.empty:
                self._logger.warning(
                    "Analitik pipeline sonrası temiz DataFrame boş. "
                    "Veri kalitesi sorunları olabilir."
                )

            anomali_str = "EVET ⚠" if summary.anomaly_detected else "hayır"
            self._logger.info(
                f"✓ Katman 3 tamamlandı: "
                f"{len(clean_df)} temiz kayıt | "
                f"avg_sentiment={summary.average_sentiment_score:+.3f} | "
                f"anomali={anomali_str}"
            )
            return clean_df, summary

        except Exception as e:
            self._logger.error(
                f"✗ Katman 3 HATA: {e}\n{traceback.format_exc()}"
            )
            raise

    def _step_report(self, clean_df, summary) -> dict:
        """
        Katman 4: Kurumsal raporlama ve dosya export.
        Çıktı: dict[format → Path]
        """
        self._logger.info(
            f"▶ KATMAN 4/4: Raporlama Başlıyor... "
            f"(formatlar={self._config.report.export_formats})"
        )
        try:
            reporter = DataReporter(config=self._config.report)
            export_paths = reporter.export_all(clean_df, summary)

            produced = sum(1 for p in export_paths.values() if p)
            self._logger.info(
                f"✓ Katman 4 tamamlandı: {produced} dosya üretildi."
            )
            return export_paths

        except Exception as e:
            self._logger.error(
                f"✗ Katman 4 HATA: {e}\n{traceback.format_exc()}"
            )
            raise

    def run(self) -> dict:
        """
        Tam pipeline'ı çalıştırır ve özet döndürür.

        Başarılı çalışmada dönen dict:
          {
            "status"           : "success",
            "records_scraped"  : int,
            "records_analyzed" : int,
            "success_rate"     : float,
            "export_paths"     : dict[str, str],
            "elapsed_seconds"  : float,
            "anomaly_detected" : bool,
          }
        """
        logger = logging.getLogger(__name__)

        logger.info("=" * 60)
        logger.info(f"  {self._config.app_name} v{self._config.app_version}")
        logger.info(f"  AI Provider  : {self._config.ai.provider.upper()}")
        logger.info(f"  Veri Kaynağı : {self._input_file or self._config.data_source.source_type}")
        logger.info(f"  Max Kayıt    : {self._config.data_source.max_records}")
        logger.info(f"  Granularity  : {self._granularity}")
        logger.info(f"  Async Mod    : {'Evet' if self._async_mode else 'Hayır'}")
        logger.info("=" * 60)

        wall_start = time.time()

        # ── Katman 1: Veri Kazıma / Yükleme ──────────────────────────────────
        raw_records = self._step_scrape()

        # ── Katman 2: AI İşleme (sync veya async) ────────────────────────────
        batch_result = self._step_ai_process(raw_records)

        # ── Katman 3: Analitik ────────────────────────────────────────────────
        clean_df, summary = self._step_analytics(batch_result)

        # ── Katman 4: Raporlama ───────────────────────────────────────────────
        export_paths = self._step_report(clean_df, summary)

        # ── Özet ──────────────────────────────────────────────────────────────
        elapsed = round(time.time() - wall_start, 2)
        produced = sum(1 for p in export_paths.values() if p)

        logger.info("=" * 60)
        logger.info("  ✅ PIPELINE BAŞARIYLA TAMAMLANDI")
        logger.info(f"  Toplam süre      : {elapsed}s")
        logger.info(f"  Yüklenen kayıt   : {len(raw_records)}")
        logger.info(f"  Analiz edilen    : {len(batch_result.successful)}")
        logger.info(f"  Başarı oranı     : {batch_result.success_rate:.1%}")
        logger.info(f"  Üretilen dosya   : {produced}")
        if summary.anomaly_detected:
            logger.warning(
                f"  ⚠ ANOMALİ TESPİT EDİLDİ: "
                f"{len(summary.anomaly_details)} olay"
            )
        logger.info("=" * 60)

        return {
            "status": "success",
            "records_scraped": len(raw_records),
            "records_analyzed": len(batch_result.successful),
            "success_rate": batch_result.success_rate,
            "export_paths": {
                fmt: str(path) for fmt, path in export_paths.items() if path
            },
            "elapsed_seconds": elapsed,
            "anomaly_detected": summary.anomaly_detected,
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Program giriş noktası.

    Çıkış kodları:
      0 → Başarılı
      1 → Pipeline hatası
      2 → Yapılandırma / argüman hatası
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    # Loglama başlat
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Yapılandırmayı oluştur ve CLI argümanları ile override et
    try:
        config = AppConfig()

        if args.provider:
            config.ai.provider = args.provider
            logger.info(f"CLI override: AI provider → {args.provider}")

        if args.source:
            config.data_source.source_type = args.source
            logger.info(f"CLI override: data source → {args.source}")

        if args.records:
            config.data_source.max_records = args.records
            logger.info(f"CLI override: max records → {args.records}")

        # --output ve --output-dir (geriye dönük uyumluluk)
        output_target = args.output or args.output_dir
        if output_target:
            # Hem dizin hem dosya tabanı olarak kullan
            config.report.output_dir = str(Path(output_target).parent)
            logger.info(f"CLI override: output dir → {config.report.output_dir}")

        if args.no_excel and "xlsx" in config.report.export_formats:
            config.report.export_formats.remove("xlsx")
            logger.info("Excel export devre dışı bırakıldı.")

        # --input dosyası varsa doğrula
        if args.input and not Path(args.input).exists():
            logger.error(f"Girdi dosyası bulunamadı: {args.input}")
            return 2

        # Yapılandırmayı doğrula (fail-fast)
        config.validate()

    except EnvironmentError as e:
        logger.error(f"Yapılandırma hatası: {e}")
        return 2
    except Exception as e:
        logger.error(f"Başlatma hatası: {e}\n{traceback.format_exc()}")
        return 2

    # Pipeline'ı çalıştır
    try:
        orchestrator = DataPipelineOrchestrator(
            config=config,
            input_file=args.input,
            granularity=args.granularity,
            async_mode=args.async_mode,
            max_concurrent=args.max_concurrent,
        )
        orchestrator.run()
        return 0

    except (ScraperError, AIProcessorError) as e:
        logger.error(f"Pipeline hatası: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Kullanıcı tarafından durduruldu (Ctrl+C).")
        return 1
    except Exception as e:
        logger.critical(f"Kritik hata: {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
