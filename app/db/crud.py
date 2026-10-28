"""
app/db/crud.py
==============
CRUD helpers for invoice_records and correction_log.
All DB interaction in the app should go through these functions —
never raw SQL scattered in node code.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import CorrectionLogORM, InvoiceRecordORM
from app.nodes.schemas import InvoiceRecord

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _record_to_orm(record: InvoiceRecord) -> InvoiceRecordORM:
    extraction_json = None
    if record.extraction is not None:
        extraction_json = record.extraction.model_dump(mode="json")

    return InvoiceRecordORM(
        id=record.id,
        document_id=record.document_id,
        file_hash=record.file_hash,
        original_filename=record.original_filename,
        raw_file_path=record.raw_file_path,
        processed_image_paths=record.processed_image_paths,
        extraction_json=extraction_json,
        raw_llm_response=record.raw_llm_response,
        status=record.status,
        validation_errors=record.validation_errors,
        low_confidence_fields=record.low_confidence_fields,
        is_duplicate=record.is_duplicate,
        duplicate_of=record.duplicate_of,
        retry_count=record.retry_count,
        needs_human_review=record.needs_human_review,
        human_reviewed=record.human_reviewed,
        reviewed_by=record.reviewed_by,
        reviewed_at=record.reviewed_at,
        reviewer_notes=record.reviewer_notes,
        export_csv_path=record.export_csv_path,
        export_tally_xml_path=record.export_tally_xml_path,
    )


def _orm_to_record(orm: InvoiceRecordORM) -> InvoiceRecord:
    from app.nodes.schemas import InvoiceExtraction

    extraction = None
    if orm.extraction_json:
        try:
            extraction = InvoiceExtraction.model_validate(orm.extraction_json)
        except Exception as exc:
            logger.warning("Failed to deserialise extraction for %s: %s", orm.document_id, exc)

    return InvoiceRecord(
        id=orm.id,
        document_id=orm.document_id,
        file_hash=orm.file_hash,
        original_filename=orm.original_filename,
        raw_file_path=orm.raw_file_path,
        processed_image_paths=orm.processed_image_paths or [],
        extraction=extraction,
        raw_llm_response=orm.raw_llm_response,
        status=orm.status,
        validation_errors=orm.validation_errors or [],
        low_confidence_fields=orm.low_confidence_fields or [],
        is_duplicate=orm.is_duplicate,
        duplicate_of=orm.duplicate_of,
        retry_count=orm.retry_count,
        needs_human_review=orm.needs_human_review,
        human_reviewed=orm.human_reviewed,
        reviewed_by=orm.reviewed_by,
        reviewed_at=orm.reviewed_at,
        reviewer_notes=orm.reviewer_notes,
        export_csv_path=orm.export_csv_path,
        export_tally_xml_path=orm.export_tally_xml_path,
        created_at=orm.created_at or datetime.utcnow(),
        updated_at=orm.updated_at or datetime.utcnow(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CRUD functions
# ──────────────────────────────────────────────────────────────────────────────

def create_invoice_record(session: Session, record: InvoiceRecord) -> InvoiceRecord:
    """Insert a new invoice record. Raises if ID already exists."""
    orm = _record_to_orm(record)
    session.add(orm)
    session.flush()
    logger.info("Created invoice record %s (doc_id=%s)", record.id, record.document_id)
    return record


def get_invoice_record(session: Session, record_id: str) -> Optional[InvoiceRecord]:
    orm = session.get(InvoiceRecordORM, record_id)
    return _orm_to_record(orm) if orm else None


def get_by_document_id(session: Session, document_id: str) -> Optional[InvoiceRecord]:
    orm = (
        session.query(InvoiceRecordORM)
        .filter(InvoiceRecordORM.document_id == document_id)
        .first()
    )
    return _orm_to_record(orm) if orm else None


def update_invoice_record(session: Session, record: InvoiceRecord) -> InvoiceRecord:
    """Upsert (merge) an invoice record."""
    orm_existing = session.get(InvoiceRecordORM, record.id)
    if orm_existing is None:
        return create_invoice_record(session, record)

    # Update mutable fields
    new_orm = _record_to_orm(record)
    for col in InvoiceRecordORM.__table__.columns:
        if col.name in ("id", "created_at"):
            continue
        setattr(orm_existing, col.name, getattr(new_orm, col.name))

    session.flush()
    logger.info("Updated invoice record %s → status=%s", record.id, record.status)
    return record


def find_duplicates(
    session: Session,
    vendor_name: str,
    invoice_number: str,
    grand_total: float,
    invoice_date: object,
    exclude_id: str | None = None,
) -> List[InvoiceRecord]:
    """
    Duplicate detection: match on (vendor_name ILIKE, invoice_number, grand_total, date).
    Returns existing records that match.
    """
    query = (
        session.query(InvoiceRecordORM)
        .filter(
            InvoiceRecordORM.extraction_json["invoice_number"].astext == invoice_number,
        )
    )
    if exclude_id:
        query = query.filter(InvoiceRecordORM.document_id != exclude_id)

    candidates = query.all()
    results = []
    for orm in candidates:
        if orm.extraction_json is None:
            continue
        stored_total = orm.extraction_json.get("tax_breakup", {}).get("grand_total")
        stored_vendor = orm.extraction_json.get("vendor_name", "")
        stored_date = orm.extraction_json.get("invoice_date")
        if (
            abs(float(stored_total or 0) - grand_total) < 1.0
            and stored_vendor.lower().strip() == vendor_name.lower().strip()
            and str(stored_date) == str(invoice_date)
        ):
            results.append(_orm_to_record(orm))

    return results


def log_correction(
    session: Session,
    document_id: str,
    field_name: str,
    original_value: str | None,
    corrected_value: str | None,
    reviewed_by: str | None = None,
) -> None:
    """Persist one human correction for evaluation/training purposes."""
    entry = CorrectionLogORM(
        document_id=document_id,
        field_name=field_name,
        original_value=str(original_value) if original_value is not None else None,
        corrected_value=str(corrected_value) if corrected_value is not None else None,
        reviewed_by=reviewed_by,
    )
    session.add(entry)
    logger.debug(
        "Logged correction: doc=%s field=%s '%s' → '%s'",
        document_id, field_name, original_value, corrected_value,
    )


def list_needs_review(session: Session, limit: int = 50) -> List[InvoiceRecord]:
    """Return the oldest N records pending human review."""
    orms = (
        session.query(InvoiceRecordORM)
        .filter(
            InvoiceRecordORM.needs_human_review == True,
            InvoiceRecordORM.human_reviewed == False,
            InvoiceRecordORM.status == "NEEDS_REVIEW",
        )
        .order_by(InvoiceRecordORM.created_at.asc())
        .limit(limit)
        .all()
    )
    return [_orm_to_record(o) for o in orms]
