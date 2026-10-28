"""
app/db/models.py
=================
SQLAlchemy ORM models for Postgres.

Tables
------
invoice_records   – one row per uploaded document
correction_log    – every human edit for Phase 6 evaluation training data
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class InvoiceRecordORM(Base):
    """Mirrors app.nodes.schemas.InvoiceRecord."""

    __tablename__ = "invoice_records"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), nullable=False, index=True)
    file_hash = Column(String(64), nullable=False, index=True)   # SHA-256, dedup key
    original_filename = Column(String(512), nullable=False)
    raw_file_path = Column(String(1024), nullable=False)
    processed_image_paths = Column(JSON, nullable=False, default=list)  # list[str]

    # LLM output
    extraction_json = Column(JSON, nullable=True)   # serialised InvoiceExtraction
    raw_llm_response = Column(Text, nullable=True)  # raw LLM JSON string

    # Processing state
    status = Column(String(32), nullable=False, default="PROCESSING")
    validation_errors = Column(JSON, nullable=False, default=list)   # list[str]
    low_confidence_fields = Column(JSON, nullable=False, default=list)
    is_duplicate = Column(Boolean, nullable=False, default=False)
    duplicate_of = Column(String(36), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    needs_human_review = Column(Boolean, nullable=False, default=False)

    # Human review
    human_reviewed = Column(Boolean, nullable=False, default=False)
    reviewed_by = Column(String(256), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    # Export
    export_csv_path = Column(String(1024), nullable=True)
    export_tally_xml_path = Column(String(1024), nullable=True)

    # Timestamps (stored in UTC)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class CorrectionLogORM(Base):
    """
    Labelled human corrections — used in Phase 6 for per-field accuracy evaluation
    and as future fine-tuning / RAG training data.
    """

    __tablename__ = "correction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(36), nullable=False, index=True)
    field_name = Column(String(128), nullable=False)        # e.g. "vendor_gstin"
    original_value = Column(Text, nullable=True)             # what the LLM extracted
    corrected_value = Column(Text, nullable=True)            # what the human changed it to
    correction_source = Column(String(64), default="human_review")
    reviewed_by = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
