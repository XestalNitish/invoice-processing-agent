"""
app/main.py
============
FastAPI backend service for the Indian GST Invoice Processing Agent.

Endpoints:
- GET  /health                     : Service health, DB connectivity, Local LLM status
- POST /api/v1/invoices/upload     : Upload invoice (PDF/Image), preprocess, extract, validate
- GET  /api/v1/invoices/{doc_id}   : Retrieve specific invoice record
- GET  /api/v1/invoices            : List invoices (with status filter)
- POST /api/v1/invoices/{doc_id}/review : Submit human corrections / approval
- GET  /api/v1/models              : Check available LLM models & active provider
- POST /api/v1/eval/run            : Run evaluation benchmark and return accuracy/latency metrics
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.db.crud import (
    create_invoice_record,
    get_by_document_id,
    list_needs_review,
    log_correction,
    update_invoice_record,
)
from app.db.models import InvoiceRecordORM
from app.db.session import get_session, init_db
from app.nodes.schemas import InvoiceExtraction, InvoiceRecord
from app.nodes.validators import validate_invoice
from app.pipeline_phase1 import run_phase1
from local_llm.local_llm_service import is_local_capable, is_server_alive, start_local_server

logger = logging.getLogger("invoice_agent_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: DB initialization and Local LLM warm-up
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Invoice Agent API server...")
    
    # 1. Initialize SQLite / Postgres DB
    try:
        init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as exc:
        logger.warning("DB initialization deferred/skipped: %s", exc)

    # 2. Check local LLM if configured
    if settings.llm_provider == "local":
        if not is_server_alive():
            logger.info("Local LLM provider active. Launching local inference server...")
            try:
                start_local_server(port=settings.local_llm_port, wait_timeout=30)
            except Exception as e:
                logger.warning("Could not auto-start local LLM: %s", e)
        else:
            logger.info("Local inference server is already running and ready.")

    yield
    logger.info("Shutting down Invoice Agent API server.")


app = FastAPI(
    title="Invoice Processing Agent API",
    description="Intelligent Indian GST Invoice Extraction, Deterministic Validation, and Tally/CSV Export Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

class ReviewSubmission(BaseModel):
    decision: str = Field(..., description="'APPROVED' or 'REJECTED'")
    reviewed_by: str = Field("human_reviewer", description="Name/ID of reviewer")
    notes: Optional[str] = Field(None, description="Optional reviewer remarks")
    corrected_data: Optional[Dict[str, Any]] = Field(None, description="Corrected invoice fields if modified")


class HealthResponse(BaseModel):
    status: str
    active_provider: str
    local_llm_online: bool
    local_llm_capable: bool
    database_connected: bool
    timestamp: str


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "service": "Indian GST Invoice Processing Agent",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    settings = get_settings()
    db_ok = False
    try:
        with get_session() as s:
            s.execute(InvoiceRecordORM.__table__.select().limit(1))
            db_ok = True
    except Exception:
        db_ok = False

    return HealthResponse(
        status="HEALTHY",
        active_provider=settings.llm_provider,
        local_llm_online=is_server_alive(),
        local_llm_capable=is_local_capable(),
        database_connected=db_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/invoices/upload", tags=["Invoices"])
async def upload_invoice(file: UploadFile = File(...)):
    """
    Upload an invoice image (PNG, JPG) or PDF.
    Runs preprocessing, LLM extraction, and deterministic validation.
    """
    settings = get_settings()
    doc_id = str(uuid.uuid4())
    
    # Validate extension
    ext = Path(file.filename or "file.pdf").suffix.lower()
    allowed_exts = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(allowed_exts))}",
        )

    # Save to raw storage
    raw_dir = Path(settings.raw_images_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_path = raw_dir / f"{doc_id}_{file.filename}"
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Phase 1: Preprocess + Extract + Hash
        record = run_phase1(file_path=temp_path, document_id=doc_id, persist_to_db=False)
        
        # Phase 2: Deterministic Validation
        if record.extraction:
            val_result = validate_invoice(
                extraction=record.extraction,
                confidence_threshold=settings.confidence_review_threshold,
                is_duplicate=record.is_duplicate,
            )
            record.validation_errors.extend(val_result.errors)
            record.low_confidence_fields.extend(val_result.low_confidence_fields)
            record.needs_human_review = val_result.needs_review
            record.status = "APPROVED" if val_result.is_valid and not val_result.needs_review else "NEEDS_REVIEW"
        else:
            record.status = "FAILED"
            record.needs_human_review = True

        # Persist to database if available
        try:
            with get_session() as session:
                create_invoice_record(session, record)
        except Exception as db_err:
            logger.warning("DB persist skipped: %s", db_err)

        return {
            "document_id": record.document_id,
            "status": record.status,
            "needs_human_review": record.needs_human_review,
            "validation_errors": record.validation_errors,
            "low_confidence_fields": record.low_confidence_fields,
            "extracted_data": record.extraction.model_dump() if record.extraction else None,
            "processed_images": record.processed_image_paths,
        }

    except Exception as exc:
        logger.error("Processing failed for doc_id=%s: %s", doc_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invoice processing failed: {str(exc)}",
        )


@app.get("/api/v1/invoices/{document_id}", tags=["Invoices"])
def get_invoice(document_id: str):
    """Retrieve full status, extraction, and validation state for an invoice."""
    try:
        with get_session() as session:
            record = get_by_document_id(session, document_id)
            if not record:
                raise HTTPException(status_code=404, detail="Invoice record not found.")
            return record.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/invoices", tags=["Invoices"])
def list_invoices(
    status_filter: Optional[str] = Query(None, description="NEEDS_REVIEW, APPROVED, REJECTED, PROCESSING"),
    limit: int = Query(50, ge=1, le=200)
):
    """List invoices with optional status filter."""
    try:
        with get_session() as session:
            if status_filter == "NEEDS_REVIEW":
                records = list_needs_review(session, limit=limit)
            else:
                query = session.query(InvoiceRecordORM)
                if status_filter:
                    query = query.filter(InvoiceRecordORM.status == status_filter)
                query = query.order_by(InvoiceRecordORM.created_at.desc()).limit(limit)
                from app.db.crud import _orm_to_record
                records = [_orm_to_record(o) for o in query.all()]
            return [r.model_dump() for r in records]
    except Exception as e:
        return {"records": [], "note": f"DB query fallback: {str(e)}"}


@app.post("/api/v1/invoices/{document_id}/review", tags=["Review"])
def submit_review(document_id: str, review: ReviewSubmission):
    """
    Submit a human review decision (approve or reject) and record any field edits.
    Edits are logged to the correction_log table for model evaluation.
    """
    try:
        with get_session() as session:
            record = get_by_document_id(session, document_id)
            if not record:
                raise HTTPException(status_code=404, detail="Invoice record not found.")

            # Record human edits if any
            if review.corrected_data and record.extraction:
                old_dict = record.extraction.model_dump()
                for k, v in review.corrected_data.items():
                    if k in old_dict and old_dict[k] != v:
                        log_correction(
                            session=session,
                            document_id=document_id,
                            field_name=k,
                            original_value=str(old_dict[k]),
                            corrected_value=str(v),
                            reviewed_by=review.reviewed_by,
                        )
                # Update extraction with corrected data
                try:
                    merged = {**old_dict, **review.corrected_data}
                    record.extraction = InvoiceExtraction.model_validate(merged)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid corrected data: {exc}")

            record.status = review.decision.upper()
            record.human_reviewed = True
            record.reviewed_by = review.reviewed_by
            record.reviewed_at = datetime.now(timezone.utc)
            record.reviewer_notes = review.notes
            record.needs_human_review = False

            update_invoice_record(session, record)
            return {
                "message": f"Invoice {document_id} marked as {record.status}",
                "document_id": document_id,
                "status": record.status,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models", tags=["Models"])
def list_available_models():
    """List available LLM providers and local model status."""
    settings = get_settings()
    return {
        "active_provider": settings.llm_provider,
        "vision_model": settings.vision_model,
        "local_model": {
            "capable": is_local_capable(),
            "server_running": is_server_alive(),
            "port": settings.local_llm_port,
            "weights_found": is_local_capable(),
        },
        "available_providers": ["local", "anthropic", "openai", "gemini"]
    }
