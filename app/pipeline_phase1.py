"""
app/pipeline_phase1.py
=======================
Phase 1 top-level orchestrator: preprocess → extract → store.

This is a *synchronous* runner that can be called directly or will later be
wrapped by the LangGraph workflow (Phase 3).
It does NOT run validators (Phase 2) — those come next.

Usage
-----
    from app.pipeline_phase1 import run_phase1

    result = run_phase1("path/to/invoice.pdf")
    print(result.status, result.extraction.vendor_name)
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.db.crud import create_invoice_record, find_duplicates
from app.db.session import get_session
from app.extraction.extractor import ExtractionError, extract_invoice
from app.nodes.schemas import InvoiceRecord
from app.preprocessing.image_processor import PreprocessingError, preprocess_document

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _copy_to_raw_storage(src: Path, raw_dir: Path, doc_id: str) -> Path:
    """Copy the original file into storage/raw_images/ under the doc_id prefix."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{doc_id}_{src.name}"
    shutil.copy2(src, dest)
    return dest


def run_phase1(
    file_path: str | Path,
    document_id: str | None = None,
    persist_to_db: bool = True,
) -> InvoiceRecord:
    """
    Execute Phase 1: preprocess → extract → (optional) persist.

    Parameters
    ----------
    file_path :
        Path to the invoice file (.pdf / .jpg / .png / .tiff).
    document_id :
        Optional stable ID. Auto-generated if not provided.
    persist_to_db :
        Set False in unit tests to skip DB writes.

    Returns
    -------
    InvoiceRecord
        Populated with extraction result, status, and storage paths.
        status = "NEEDS_VALIDATION" after this phase (validators not run yet).
    """
    settings = get_settings()
    file_path = Path(file_path)
    doc_id = document_id or str(uuid.uuid4())

    logger.info("=== Phase 1 START: doc_id=%s file=%s ===", doc_id, file_path.name)

    # ── Step 1: Hash + copy raw file ──────────────────────────────────────
    file_hash = _hash_file(file_path)
    raw_dest = _copy_to_raw_storage(file_path, Path(settings.raw_images_dir), doc_id)
    logger.info("File hash=%s, stored at %s", file_hash[:12] + "…", raw_dest)

    # ── Step 2: Preprocess (deskew, contrast, binarise) ──────────────────
    try:
        processed = preprocess_document(file_path, document_id=doc_id)
    except PreprocessingError as exc:
        logger.error("Preprocessing failed: %s", exc)
        record = InvoiceRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            file_hash=file_hash,
            original_filename=file_path.name,
            raw_file_path=str(raw_dest),
            status="FAILED",
            validation_errors=[f"Preprocessing error: {exc}"],
        )
        if persist_to_db:
            with get_session() as session:
                create_invoice_record(session, record)
        return record

    logger.info(
        "Preprocessing done: %d page(s), saved to %s",
        processed.page_count,
        processed.saved_paths,
    )

    # ── Step 3: Extract via Vision LLM ───────────────────────────────────
    try:
        extraction, raw_llm = extract_invoice(processed.page_base64_images)
    except ExtractionError as exc:
        logger.error("Extraction failed: %s", exc)
        record = InvoiceRecord(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            file_hash=file_hash,
            original_filename=file_path.name,
            raw_file_path=str(raw_dest),
            processed_image_paths=processed.saved_paths,
            status="FAILED",
            validation_errors=[f"Extraction error: {exc}"],
        )
        if persist_to_db:
            with get_session() as session:
                create_invoice_record(session, record)
        return record

    # ── Step 4: Lightweight duplicate pre-check ───────────────────────────
    is_dup = False
    dup_of = None
    if persist_to_db:
        with get_session() as session:
            dupes = find_duplicates(
                session,
                vendor_name=extraction.vendor_name,
                invoice_number=extraction.invoice_number,
                grand_total=extraction.tax_breakup.grand_total,
                invoice_date=extraction.invoice_date,
                exclude_id=doc_id,
            )
            if dupes:
                is_dup = True
                dup_of = dupes[0].document_id
                logger.warning(
                    "Possible duplicate: invoice_no=%s already in doc_id=%s",
                    extraction.invoice_number, dup_of,
                )

    # ── Step 5: Assemble record ───────────────────────────────────────────
    record = InvoiceRecord(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        file_hash=file_hash,
        original_filename=file_path.name,
        raw_file_path=str(raw_dest),
        processed_image_paths=processed.saved_paths,
        extraction=extraction,
        raw_llm_response=raw_llm,
        status="NEEDS_VALIDATION",  # Phase 2 validators not run yet
        is_duplicate=is_dup,
        duplicate_of=dup_of,
        needs_human_review=is_dup,  # duplicates always go to review
    )

    # ── Step 6: Persist to DB ─────────────────────────────────────────────
    if persist_to_db:
        with get_session() as session:
            create_invoice_record(session, record)
        logger.info("Persisted record id=%s to DB", record.id)

    logger.info("=== Phase 1 DONE: doc_id=%s status=%s ===", doc_id, record.status)
    return record
