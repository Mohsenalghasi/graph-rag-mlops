from __future__ import annotations

import argparse
import json
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from services.storage import StorageService
from services.extractor import PDFExtractor
from services.cleaner import TextCleaner
from services.chunker import Chunker


# -----------------------------
# Utilities
# -----------------------------
def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mb(nbytes: int) -> float:
    return nbytes / 1_048_576


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"processed_sha256": [], "quarantined_sha256": []}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        # corrupted state should not brick ingestion
        return {"processed_sha256": [], "quarantined_sha256": []}


def save_state(state_path: Path, processed_set: set[str], quarantined_set: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    payload = {
        "processed_sha256": sorted(processed_set),
        "quarantined_sha256": sorted(quarantined_set),
        "updated_at": iso_utc_now(),
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def get_page_num(page: dict[str, Any]) -> int:
    # Support either key, fallback to 0
    v = page.get("page_num", page.get("page", 0))
    try:
        return int(v or 0)
    except Exception:
        return 0


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class IngestConfig:
    # batch
    doc_type: str = "manual"
    batch_size: int = 1
    batch_max_mb: int = 50

    # file-level guards
    max_pdf_mb: int = 50
    pdf_timeout_sec: int = 60

    # scanned detection heuristics
    scanned_empty_ratio: float = 0.7
    sample_pages_for_scan_check: int = 8
    sample_min_avg_chars: int = 25  # too low => likely scanned/diagram

    # output behavior
    archive_processed: bool = True
    resume: bool = True
    state_path: Path | None = None

    # page/chunk guards
    max_page_chars: int = 200_000
    max_clean_chars: int = 200_000
    max_chunks_per_page: int = 400
    max_total_chunks_per_pdf: int = 10_000

    # debug
    debug: bool = False
    debug_char_threshold: int = 50_000


# -----------------------------
# Pipeline
# -----------------------------
class IngestionPipeline:
    def __init__(
        self,
        storage: StorageService,
        extractor: PDFExtractor,
        cleaner: TextCleaner,
        chunker: Chunker,
        config: IngestConfig,
    ):
        self.storage = storage
        self.extractor = extractor
        self.cleaner = cleaner
        self.chunker = chunker
        self.cfg = config

        self.max_pdf_bytes = int(self.cfg.max_pdf_mb) * 1024 * 1024

        # folders
        self.data_dir: Path = storage.data_dir
        self.archive_dir: Path = self.data_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.state_path: Path = self.cfg.state_path or (self.data_dir / "state" / "ingest_state.json")

    def _archive_pdf(self, pdf: Path) -> None:
        dest = self.archive_dir / pdf.name
        if dest.exists():
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = self.archive_dir / f"{pdf.stem}.{ts}{pdf.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(pdf), str(dest))

    def _quarantine(
        self,
        pdf: Path,
        reason: str,
        stats: dict,
        file_hash: str,
        processed_set: set[str],
        quarantined_set: set[str],
    ) -> None:
        clean_reason = (reason or "").strip() or "Unknown quarantine reason"
        self.storage.quarantine_file(pdf, clean_reason)
        stats["quarantined"] += 1
        quarantined_set.add(file_hash)
        if self.cfg.resume:
            save_state(self.state_path, processed_set, quarantined_set)
        print(f"[INGEST] Quarantined: {clean_reason}", flush=True)

    def _select_batch(self, all_pdfs: list[Path]) -> tuple[list[Path], int]:
        """
        Select up to batch_size PDFs, while keeping total selected bytes <= batch_max_mb (unless first file exceeds).
        """
        batch_max_bytes = int(self.cfg.batch_max_mb) * 1024 * 1024
        selected: list[Path] = []
        total_bytes = 0

        for pdf in all_pdfs:
            if len(selected) >= int(self.cfg.batch_size):
                break

            try:
                size = pdf.stat().st_size
            except Exception:
                # if can't stat, still try it (will likely fail later and quarantine)
                size = 0

            # If we already have at least one file selected and adding this exceeds MB cap -> stop
            if selected and (total_bytes + size > batch_max_bytes):
                break

            selected.append(pdf)
            total_bytes += size

        return selected, total_bytes

    def run(self) -> dict:
        all_pdfs = list(self.storage.list_pdfs())

        # resume state
        if self.cfg.resume:
            st = load_state(self.state_path)
            processed_set = set(st.get("processed_sha256", []))
            quarantined_set = set(st.get("quarantined_sha256", []))
        else:
            processed_set, quarantined_set = set(), set()

        # batch selection
        selected, selected_bytes = self._select_batch(all_pdfs)

        stats = {
            "found_total": len(all_pdfs),
            "selected_batch": len(selected),
            "batch_size": int(self.cfg.batch_size),
            "batch_max_mb": int(self.cfg.batch_max_mb),
            "selected_total_mb": round(mb(selected_bytes), 2),
            "processed": 0,
            "quarantined": 0,
            "skipped_already_done": 0,
            "chunks": 0,
            "started_at": iso_utc_now(),
        }

        if not selected:
            stats["finished_at"] = iso_utc_now()
            return stats

        print(
            f"[INGEST] Found {len(all_pdfs)} PDFs. Running batch of {len(selected)} "
            f"(limit: {self.cfg.batch_size} files, {self.cfg.batch_max_mb} MB).",
            flush=True,
        )

        for i, pdf in enumerate(selected, start=1):
            pdf_start = perf_counter()
            try:
                size_bytes = pdf.stat().st_size
                print(f"\n[INGEST] ({i}/{len(selected)}) Start: {pdf.name} ({mb(size_bytes):.1f} MB)", flush=True)

                file_hash = self.storage.sha256(pdf)

                if self.cfg.resume and (file_hash in processed_set or file_hash in quarantined_set):
                    print(f"[INGEST] Skip already handled: sha256={file_hash[:12]}...", flush=True)
                    stats["skipped_already_done"] += 1
                    continue

                # file size guard
                if size_bytes > self.max_pdf_bytes:
                    self._quarantine(
                        pdf,
                        f"File too large > {self.cfg.max_pdf_mb}MB",
                        stats,
                        file_hash,
                        processed_set,
                        quarantined_set,
                    )
                    continue

                # extraction
                t_ext = perf_counter()
                try:
                    pages = self.extractor.extract_pages(pdf)
                except MemoryError:
                    self._quarantine(pdf, "MemoryError during extraction", stats, file_hash, processed_set, quarantined_set)
                    continue
                except Exception as e:
                    self._quarantine(pdf, f"Extraction error: {type(e).__name__}: {e}", stats, file_hash, processed_set, quarantined_set)
                    continue

                if not pages:
                    self._quarantine(pdf, "No pages extracted", stats, file_hash, processed_set, quarantined_set)
                    continue

                # scanned/diagram heuristic
                sample_pages = pages[: max(1, int(self.cfg.sample_pages_for_scan_check))]
                sample_lens = [len((p.get("text") or "").strip()) for p in sample_pages]
                sample_empty = sum(1 for L in sample_lens if L == 0)
                sample_ratio = sample_empty / max(1, len(sample_pages))
                avg_chars = int(sum(sample_lens) / max(1, len(sample_lens)))

                full_empty = sum(1 for p in pages if not (p.get("text") or "").strip())
                full_ratio = full_empty / max(1, len(pages))

                print(
                    f"[INGEST] Extracted {len(pages)} pages in {perf_counter()-t_ext:.1f}s "
                    f"(sample empty~{sample_ratio:.2f}, sample avg chars~{avg_chars}, full empty={full_ratio:.2f})",
                    flush=True,
                )

                if (
                    sample_ratio > self.cfg.scanned_empty_ratio
                    or full_ratio > self.cfg.scanned_empty_ratio
                    or avg_chars < self.cfg.sample_min_avg_chars
                ):
                    self._quarantine(pdf, "Likely scanned/diagram PDF (OCR required)", stats, file_hash, processed_set, quarantined_set)
                    continue

                # timeout guard before write loop
                if perf_counter() - pdf_start > self.cfg.pdf_timeout_sec:
                    self._quarantine(pdf, f"Timeout before writing ({self.cfg.pdf_timeout_sec}s)", stats, file_hash, processed_set, quarantined_set)
                    continue

                now = iso_utc_now()
                staging_path = self.storage.staging / f"{pdf.stem}.pages.jsonl"
                curated_path = self.storage.curated / f"{pdf.stem}.chunks.jsonl"

                written_chunks = 0
                doc_chunk_counter = 0  # global per-PDF

                try:
                    with staging_path.open("w", encoding="utf-8") as stg, curated_path.open("w", encoding="utf-8") as cur:
                        for page in pages:
                            if perf_counter() - pdf_start > self.cfg.pdf_timeout_sec:
                                raise TimeoutError(f"PDF time limit exceeded ({self.cfg.pdf_timeout_sec}s)")

                            page_num = get_page_num(page)
                            raw_text = page.get("text") or ""

                            if self.cfg.debug and len(raw_text) > self.cfg.debug_char_threshold:
                                print(f"[INGEST] DEBUG: page {page_num} raw chars={len(raw_text)}", flush=True)

                            if len(raw_text) > self.cfg.max_page_chars:
                                raise MemoryError(f"Page {page_num} raw_text too large ({len(raw_text)} chars)")

                            clean_text = self.cleaner.clean(raw_text)

                            if self.cfg.debug and len(clean_text) > self.cfg.debug_char_threshold:
                                print(f"[INGEST] DEBUG: page {page_num} clean chars={len(clean_text)}", flush=True)

                            if len(clean_text) > self.cfg.max_clean_chars:
                                raise MemoryError(f"Page {page_num} clean_text too large ({len(clean_text)} chars)")

                            if not clean_text.strip():
                                continue

                            # staging
                            stg_row = {
                                "doc_id": pdf.stem,
                                "source": pdf.name,
                                "source_path": str(pdf).replace("\\", "/"),
                                "sha256": file_hash,
                                "doc_type": self.cfg.doc_type,
                                "ingested_at": now,
                                "page": page_num,
                                "text": clean_text,
                            }
                            stg.write(json.dumps(stg_row, ensure_ascii=False) + "\n")

                            # chunk
                            chunks = self.chunker.chunk(clean_text)

                            if len(chunks) > self.cfg.max_chunks_per_page:
                                raise MemoryError(f"Page {page_num} produced too many chunks ({len(chunks)} > {self.cfg.max_chunks_per_page})")

                            for j, chunk in enumerate(chunks):
                                if perf_counter() - pdf_start > self.cfg.pdf_timeout_sec:
                                    raise TimeoutError(f"PDF time limit exceeded ({self.cfg.pdf_timeout_sec}s)")

                                if written_chunks >= self.cfg.max_total_chunks_per_pdf:
                                    raise MemoryError(f"PDF produced too many chunks (> {self.cfg.max_total_chunks_per_pdf})")

                                rec = {
                                    "id": f"{pdf.stem}_p{page_num}_c{j}",
                                    "doc_id": pdf.stem,
                                    "doc_chunk_index": doc_chunk_counter,
                                    "page_chunk_index": j,
                                    "content": chunk,
                                    "content_length": len(chunk),
                                    "source": pdf.name,
                                    "source_path": str(pdf).replace("\\", "/"),
                                    "page_start": page_num,
                                    "page_end": page_num,
                                    "doc_type": self.cfg.doc_type,
                                    "ingested_at": now,
                                    "sha256": file_hash,
                                }
                                cur.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                written_chunks += 1
                                doc_chunk_counter += 1
                                stats["chunks"] += 1

                    if written_chunks == 0:
                        try:
                            staging_path.unlink(missing_ok=True)
                            curated_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        self._quarantine(pdf, "No usable pages/chunks produced", stats, file_hash, processed_set, quarantined_set)
                        continue

                except MemoryError as e:
                    print(f"\n[CRITICAL DEBUG] MemoryError on {pdf.name}!", flush=True)
                    traceback.print_exc()
                    msg = str(e).strip() or "MemoryError (no message)"
                    try:
                        staging_path.unlink(missing_ok=True)
                        curated_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._quarantine(pdf, f"Memory guard triggered: {msg}", stats, file_hash, processed_set, quarantined_set)
                    continue

                except TimeoutError as e:
                    msg = str(e).strip() or f"Timeout ({self.cfg.pdf_timeout_sec}s)"
                    try:
                        staging_path.unlink(missing_ok=True)
                        curated_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._quarantine(pdf, f"Timeout: {msg}", stats, file_hash, processed_set, quarantined_set)
                    continue

                except Exception as e:
                    try:
                        staging_path.unlink(missing_ok=True)
                        curated_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._quarantine(pdf, f"Ingestion error: {type(e).__name__}: {e}", stats, file_hash, processed_set, quarantined_set)
                    continue

                # success: archive
                if self.cfg.archive_processed:
                    self._archive_pdf(pdf)

                stats["processed"] += 1
                processed_set.add(file_hash)
                if self.cfg.resume:
                    save_state(self.state_path, processed_set, quarantined_set)

                print(f"[INGEST] SUCCESS: {pdf.name} → {written_chunks} chunks", flush=True)

            except KeyboardInterrupt:
                print("\n[INGEST] Interrupted by user (Ctrl+C). Exiting safely.", flush=True)
                stats["finished_at"] = iso_utc_now()
                return stats

        stats["finished_at"] = iso_utc_now()
        return stats


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe batch PDF ingestion: extract -> clean -> chunk -> JSONL")

    p.add_argument("--doc-type", default="manual")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--batch-max-mb", type=int, default=50)

    p.add_argument("--max-pdf-mb", type=int, default=50)
    p.add_argument("--pdf-timeout-sec", type=int, default=60)

    p.add_argument("--scanned-empty-ratio", type=float, default=0.7)
    p.add_argument("--sample-pages", type=int, default=8)
    p.add_argument("--sample-min-avg-chars", type=int, default=25)

    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--no-resume", action="store_true")

    p.add_argument("--max-page-chars", type=int, default=200_000)
    p.add_argument("--max-clean-chars", type=int, default=200_000)
    p.add_argument("--max-chunks-per-page", type=int, default=400)
    p.add_argument("--max-total-chunks-per-pdf", type=int, default=10_000)

    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-char-threshold", type=int, default=50_000)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    storage = StorageService(data_dir=data_dir)

    cfg = IngestConfig(
        doc_type=args.doc_type,
        batch_size=args.batch_size,
        batch_max_mb=args.batch_max_mb,
        max_pdf_mb=args.max_pdf_mb,
        pdf_timeout_sec=args.pdf_timeout_sec,
        scanned_empty_ratio=args.scanned_empty_ratio,
        sample_pages_for_scan_check=args.sample_pages,
        sample_min_avg_chars=args.sample_min_avg_chars,
        archive_processed=not args.no_archive,
        resume=not args.no_resume,
        state_path=(data_dir / "state" / "ingest_state.json"),
        max_page_chars=args.max_page_chars,
        max_clean_chars=args.max_clean_chars,
        max_chunks_per_page=args.max_chunks_per_page,
        max_total_chunks_per_pdf=args.max_total_chunks_per_pdf,
        debug=args.debug,
        debug_char_threshold=args.debug_char_threshold,
    )

    pipeline = IngestionPipeline(
        storage=storage,
        extractor=PDFExtractor(),
        cleaner=TextCleaner(),
        chunker=Chunker(chunk_size_tokens=350, overlap_tokens=50),
        config=cfg,
    )

    stats = pipeline.run()
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
