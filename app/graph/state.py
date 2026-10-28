"""
app/graph/state.py
==================
TypedDict state passed between LangGraph nodes.
Every field must have a default so LangGraph can instantiate partial states.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from app.nodes.schemas import InvoiceExtraction


class InvoiceAgentState(TypedDict, total=False):
    # ── Input metadata ────────────────────────────────────────────────────
    document_id: str
    file_path: str                        # original uploaded file path
    processed_image_paths: List[str]      # paths of preprocessed page PNGs
    page_base64_images: List[str]         # base64 PNGs (for LLM API calls)
    file_hash: str                        # SHA-256 of raw bytes (dedup key)
    original_filename: str

    # ── Extraction & RAG ─────────────────────────────────────────────────
    extracted_data: Optional[InvoiceExtraction]
    raw_llm_response: Optional[str]       # raw JSON string from LLM
    rag_suggestions: Dict[str, Any]       # {"canonical_vendor", "default_ledger", "similarity"}

    # ── Validation & self-correction loop ────────────────────────────────
    is_valid: bool
    validation_errors: List[str]          # fed back to LLM as corrective prompt
    low_confidence_fields: List[str]      # highlighted in Streamlit review UI
    is_duplicate: bool
    duplicate_of: Optional[str]           # document_id of original, if duplicate
    retry_count: int

    # ── Human-in-the-loop & final output ─────────────────────────────────
    needs_human_review: bool
    human_reviewed: bool
    reviewer_notes: Optional[str]
    export_paths: Dict[str, str]          # {"tally_xml": path, "csv": path}

    # ── Overall status ────────────────────────────────────────────────────
    status: str                           # PROCESSING | NEEDS_REVIEW | COMPLETED | FAILED
    error_message: Optional[str]          # set on FAILED
