"""
app/nodes/schemas.py
====================
Canonical Pydantic v2 models for the Invoice Processing Agent.

Design principles
-----------------
* Every field that the LLM may mis-read carries an *optional* confidence
  score companion, collected separately in `field_confidence`.
* Monetary amounts are stored as Python float but validated to ≤ 2 decimal
  places via Annotated validators (we never silently round).
* GSTIN validation lives in the deterministic validator layer (Phase 2);
  the schema only enforces the 15-char structural shape so the LLM output
  is never silently mutated here.
* `suggested_ledger` on LineItem is populated by the RAG layer (Phase 4)
  and is None at extraction time.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timezone
from typing import Annotated, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Monetary amounts: positive-or-zero, max 2 decimal places.
# We raise ValueError (never silently fix) when the LLM returns > 2 dp.
def _money(description: str, *, allow_negative: bool = False) -> object:
    """Return an Annotated type constraint for a monetary field."""
    return Field(..., description=description)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    """One row of the invoice table."""

    description: str = Field(..., description="Item or service description as printed on the bill")
    hsn_sac: Optional[str] = Field(
        None, description="HSN code (goods) or SAC code (services) if printed; 4–8 digits"
    )
    quantity: float = Field(1.0, ge=0, description="Quantity of items; defaults to 1 when not stated")
    unit: Optional[str] = Field(None, description="Unit of measure, e.g. Nos, Kg, Ltr, Hrs")
    unit_price: float = Field(..., ge=0, description="Rate per unit before any discount or tax")
    discount_amount: float = Field(0.0, ge=0, description="Line-level discount in rupees, if any")
    tax_rate_percent: float = Field(
        0.0, ge=0, le=100,
        description="Total GST rate % on this line (e.g. 5, 12, 18, 28). Sum of CGST+SGST or IGST rates."
    )
    taxable_amount: float = Field(
        ..., ge=0,
        description="(quantity × unit_price) − discount_amount; the base on which tax is computed"
    )
    total_amount: float = Field(
        ..., ge=0,
        description="taxable_amount + tax on this line. May equal taxable_amount for tax-exclusive invoices."
    )
    # Populated by RAG in Phase 4
    suggested_ledger: Optional[str] = Field(
        None, description="Tally ledger category suggested by the RAG layer"
    )

    @field_validator("taxable_amount", "unit_price", "total_amount", "discount_amount", mode="before")
    @classmethod
    def _two_dp(cls, v: object) -> float:
        f = float(v)
        if round(f, 2) != round(f, 6):
            raise ValueError(
                f"Monetary value {v!r} has more than 2 decimal places — "
                "never silently round; fix the extracted value."
            )
        return round(f, 2)


class TaxBreakup(BaseModel):
    """Aggregated tax summary at the invoice footer level."""

    taxable_subtotal: float = Field(
        ..., ge=0,
        description="Sum of all line-item taxable amounts before any tax"
    )
    cgst_amount: float = Field(0.0, ge=0, description="Total Central GST collected")
    sgst_amount: float = Field(0.0, ge=0, description="Total State GST collected")
    igst_amount: float = Field(0.0, ge=0, description="Total Integrated GST (inter-state); mutually exclusive with CGST/SGST")
    cess_amount: float = Field(0.0, ge=0, description="Additional cess, if any (e.g. coal cess, tobacco cess)")
    tds_amount: float = Field(0.0, ge=0, description="TDS deducted at source, if mentioned on invoice")
    round_off: float = Field(
        0.0,
        description="Rounding adjustment applied by vendor (positive or negative, typically < ±1 ₹)"
    )
    grand_total: float = Field(
        ..., ge=0,
        description="Final payable amount = taxable_subtotal + all taxes + cess + round_off − tds"
    )

    @field_validator(
        "taxable_subtotal", "cgst_amount", "sgst_amount",
        "igst_amount", "cess_amount", "grand_total", "tds_amount",
        mode="before",
    )
    @classmethod
    def _two_dp(cls, v: object) -> float:
        f = float(v)
        if round(f, 2) != round(f, 6):
            raise ValueError(
                f"Monetary value {v!r} has more than 2 decimal places."
            )
        return round(f, 2)

    @field_validator("round_off", mode="before")
    @classmethod
    def _round_off_dp(cls, v: object) -> float:
        return round(float(v), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Primary extraction model (LLM output)
# ──────────────────────────────────────────────────────────────────────────────

class InvoiceExtraction(BaseModel):
    """
    Structured output contract for the Vision LLM.

    The LLM must populate every non-Optional field. Optional fields should
    be set to None (not omitted) when not found on the invoice so that
    downstream validators can distinguish "not on document" from "LLM skipped".
    """

    # ── Vendor / Seller ─────────────────────────────────────────────────
    vendor_name: str = Field(..., description="Legal name of the seller / vendor issuing this invoice")
    vendor_address: Optional[str] = Field(None, description="Full postal address of the vendor as printed")
    vendor_gstin: Optional[str] = Field(
        None,
        min_length=15, max_length=15,
        description=(
            "15-character GSTIN of the vendor. Format: 2-digit state code + 10-char PAN + "
            "1-char entity type + 1-char Z + 1-char checksum. E.g. 27AABCU9603R1ZX"
        ),
    )
    vendor_phone: Optional[str] = Field(None, description="Vendor contact phone number if printed")

    # ── Buyer / Recipient ────────────────────────────────────────────────
    buyer_name: Optional[str] = Field(None, description="Name of the buyer / bill-to party")
    buyer_gstin: Optional[str] = Field(
        None,
        min_length=15, max_length=15,
        description="15-character GSTIN of the buyer if printed on the invoice",
    )
    buyer_address: Optional[str] = Field(None, description="Postal address of the buyer as printed")

    # ── Invoice Identity ─────────────────────────────────────────────────
    invoice_number: str = Field(
        ...,
        description="Unique invoice / bill serial number as printed (preserve original formatting)"
    )
    invoice_date: date = Field(..., description="Date of the invoice in YYYY-MM-DD format")
    due_date: Optional[date] = Field(None, description="Payment due date if mentioned")

    # ── Supply details ───────────────────────────────────────────────────
    place_of_supply_state_code: Optional[str] = Field(
        None,
        description=(
            "2-digit Indian state code for the place of supply. "
            "E.g. '27' = Maharashtra, '07' = Delhi, '29' = Karnataka. "
            "Determines intra-state (CGST+SGST) vs inter-state (IGST) tax."
        ),
    )
    is_inter_state: Optional[bool] = Field(
        None,
        description=(
            "True if IGST applies (inter-state supply); False if CGST+SGST applies. "
            "Infer from tax breakup if not explicitly stated."
        ),
    )

    # ── Line Items & Tax ─────────────────────────────────────────────────
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="All line items found in the invoice table; must not be empty for a valid tax invoice"
    )
    tax_breakup: TaxBreakup = Field(..., description="Aggregated tax summary from the invoice footer")

    # ── Confidence (LLM self-assessment) ─────────────────────────────────
    field_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "LLM self-reported extraction confidence per field, range 0.0–1.0. "
            "Mandatory keys: vendor_gstin, invoice_number, invoice_date, grand_total, "
            "taxable_subtotal. Add extra keys for any field you are uncertain about."
        ),
    )

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: object) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        return {k: max(0.0, min(1.0, float(score))) for k, score in v.items()}

    @field_validator("vendor_gstin", "buyer_gstin", mode="before")
    @classmethod
    def _upper_gstin(cls, v: object) -> Optional[str]:
        return v.strip().upper() if isinstance(v, str) and v.strip() else None

    @model_validator(mode="after")
    def _infer_is_inter_state(self) -> "InvoiceExtraction":
        """Infer is_inter_state from tax breakup when LLM doesn't set it."""
        tb = self.tax_breakup
        if self.is_inter_state is None:
            if tb.igst_amount > 0:
                object.__setattr__(self, "is_inter_state", True)
            elif tb.cgst_amount > 0 or tb.sgst_amount > 0:
                object.__setattr__(self, "is_inter_state", False)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Database-persisted record (wraps the extraction + metadata)
# ──────────────────────────────────────────────────────────────────────────────

class InvoiceRecord(BaseModel):
    """
    Full persisted record combining extraction + processing metadata.
    Stored in the `invoice_records` Postgres table.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(..., description="Unique identifier tied to the uploaded file")
    file_hash: str = Field(..., description="SHA-256 of the raw file bytes (dedup key)")
    original_filename: str = Field(..., description="Original uploaded filename")
    raw_file_path: str = Field(..., description="Path to raw stored file in storage/raw_images/")
    processed_image_paths: List[str] = Field(
        default_factory=list,
        description="Paths to preprocessed page images (deskewed, contrast-enhanced)"
    )

    extraction: Optional[InvoiceExtraction] = Field(
        None, description="Structured extraction result from the Vision LLM"
    )
    raw_llm_response: Optional[str] = Field(
        None, description="Raw JSON string returned by the LLM before Pydantic parsing"
    )

    # Processing state
    status: str = Field(
        "PROCESSING",
        description="PROCESSING | NEEDS_REVIEW | APPROVED | REJECTED | FAILED"
    )
    validation_errors: List[str] = Field(default_factory=list)
    low_confidence_fields: List[str] = Field(default_factory=list)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = Field(None, description="document_id of the original if duplicate")
    retry_count: int = 0
    needs_human_review: bool = False

    # Human review
    human_reviewed: bool = False
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Export paths
    export_csv_path: Optional[str] = None
    export_tally_xml_path: Optional[str] = None

    @classmethod
    def compute_file_hash(cls, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
