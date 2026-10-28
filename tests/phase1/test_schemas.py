"""
tests/phase1/test_schemas.py
=============================
Unit tests for app/nodes/schemas.py (InvoiceExtraction, LineItem, TaxBreakup, InvoiceRecord).

Run with:
    pytest tests/phase1/test_schemas.py -v
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from app.nodes.schemas import (
    InvoiceExtraction,
    InvoiceRecord,
    LineItem,
    TaxBreakup,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – minimal valid data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_line_item_data():
    return {
        "description": "Office Chair",
        "quantity": 2.0,
        "unit_price": 5000.00,
        "taxable_amount": 10000.00,
        "total_amount": 11800.00,
        "tax_rate_percent": 18.0,
    }


@pytest.fixture
def valid_tax_breakup_data():
    return {
        "taxable_subtotal": 10000.00,
        "cgst_amount": 900.00,
        "sgst_amount": 900.00,
        "igst_amount": 0.00,
        "grand_total": 11800.00,
    }


@pytest.fixture
def valid_extraction_data(valid_line_item_data, valid_tax_breakup_data):
    return {
        "vendor_name": "Acme Stationery Pvt Ltd",
        "vendor_gstin": "27AABCU9603R1ZX",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-03-15",
        "line_items": [valid_line_item_data],
        "tax_breakup": valid_tax_breakup_data,
        "field_confidence": {
            "vendor_gstin": 0.95,
            "invoice_number": 0.99,
            "invoice_date": 0.98,
            "grand_total": 0.97,
            "taxable_subtotal": 0.96,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# LineItem tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLineItem:
    def test_valid_line_item(self, valid_line_item_data):
        item = LineItem(**valid_line_item_data)
        assert item.description == "Office Chair"
        assert item.quantity == 2.0
        assert item.unit_price == 5000.00
        assert item.taxable_amount == 10000.00
        assert item.total_amount == 11800.00
        assert item.suggested_ledger is None

    def test_optional_fields_default(self):
        item = LineItem(
            description="Service charge",
            unit_price=1000.0,
            taxable_amount=1000.0,
            total_amount=1000.0,
        )
        assert item.quantity == 1.0
        assert item.hsn_sac is None
        assert item.unit is None
        assert item.discount_amount == 0.0
        assert item.tax_rate_percent == 0.0

    def test_monetary_values_clamped_to_2dp(self):
        """Values with exactly 2dp should pass."""
        item = LineItem(
            description="Test",
            unit_price=99.99,
            taxable_amount=99.99,
            total_amount=99.99,
        )
        assert item.unit_price == 99.99

    def test_more_than_2dp_raises(self):
        """Three decimal places must raise ValidationError (never silently round)."""
        with pytest.raises(ValidationError, match="2 decimal places"):
            LineItem(
                description="Bad amount",
                unit_price=99.999,    # 3 dp — invalid
                taxable_amount=99.99,
                total_amount=99.99,
            )

    def test_negative_quantity_raises(self):
        with pytest.raises(ValidationError):
            LineItem(
                description="Negative qty",
                quantity=-1,
                unit_price=100.0,
                taxable_amount=100.0,
                total_amount=100.0,
            )

    def test_negative_unit_price_raises(self):
        with pytest.raises(ValidationError):
            LineItem(
                description="Credit note line",
                unit_price=-500.0,
                taxable_amount=-500.0,
                total_amount=-500.0,
            )

    def test_hsn_and_unit_fields_optional(self):
        item = LineItem(
            description="Labour",
            hsn_sac="9954",
            unit="Hrs",
            unit_price=500.0,
            taxable_amount=500.0,
            total_amount=590.0,
            tax_rate_percent=18.0,
        )
        assert item.hsn_sac == "9954"
        assert item.unit == "Hrs"


# ──────────────────────────────────────────────────────────────────────────────
# TaxBreakup tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTaxBreakup:
    def test_valid_intra_state(self, valid_tax_breakup_data):
        tb = TaxBreakup(**valid_tax_breakup_data)
        assert tb.cgst_amount == 900.00
        assert tb.sgst_amount == 900.00
        assert tb.igst_amount == 0.00

    def test_valid_inter_state(self):
        tb = TaxBreakup(
            taxable_subtotal=10000.00,
            igst_amount=1800.00,
            grand_total=11800.00,
        )
        assert tb.igst_amount == 1800.00
        assert tb.cgst_amount == 0.00
        assert tb.sgst_amount == 0.00

    def test_round_off_positive(self):
        tb = TaxBreakup(
            taxable_subtotal=1000.00,
            cgst_amount=90.00,
            sgst_amount=90.00,
            round_off=0.50,
            grand_total=1180.50,
        )
        assert tb.round_off == 0.50

    def test_round_off_negative(self):
        tb = TaxBreakup(
            taxable_subtotal=1000.00,
            cgst_amount=90.00,
            sgst_amount=90.00,
            round_off=-0.34,
            grand_total=1179.66,
        )
        assert tb.round_off == -0.34

    def test_tds_deduction(self):
        tb = TaxBreakup(
            taxable_subtotal=50000.00,
            cgst_amount=4500.00,
            sgst_amount=4500.00,
            tds_amount=5000.00,
            grand_total=54000.00,
        )
        assert tb.tds_amount == 5000.00

    def test_monetary_more_than_2dp_raises(self):
        with pytest.raises(ValidationError, match="2 decimal places"):
            TaxBreakup(
                taxable_subtotal=10000.001,
                grand_total=10000.00,
            )

    def test_negative_amounts_raise(self):
        with pytest.raises(ValidationError):
            TaxBreakup(
                taxable_subtotal=-100.00,
                grand_total=100.00,
            )

    def test_cess_field(self):
        tb = TaxBreakup(
            taxable_subtotal=100000.00,
            igst_amount=28000.00,
            cess_amount=12000.00,
            grand_total=140000.00,
        )
        assert tb.cess_amount == 12000.00


# ──────────────────────────────────────────────────────────────────────────────
# InvoiceExtraction tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInvoiceExtraction:
    def test_valid_extraction(self, valid_extraction_data):
        inv = InvoiceExtraction(**valid_extraction_data)
        assert inv.vendor_name == "Acme Stationery Pvt Ltd"
        assert inv.invoice_number == "INV-2024-001"
        assert inv.invoice_date == date(2024, 3, 15)
        assert len(inv.line_items) == 1
        assert inv.tax_breakup.grand_total == 11800.00

    def test_gstin_uppercased(self, valid_extraction_data):
        data = {**valid_extraction_data, "vendor_gstin": "27aabcu9603r1zx"}
        inv = InvoiceExtraction(**data)
        assert inv.vendor_gstin == "27AABCU9603R1ZX"  # auto-uppercased

    def test_gstin_none_when_empty_string(self, valid_extraction_data):
        data = {**valid_extraction_data, "vendor_gstin": ""}
        inv = InvoiceExtraction(**data)
        assert inv.vendor_gstin is None

    def test_gstin_none_when_whitespace(self, valid_extraction_data):
        data = {**valid_extraction_data, "vendor_gstin": "   "}
        inv = InvoiceExtraction(**data)
        assert inv.vendor_gstin is None

    def test_gstin_wrong_length_raises(self, valid_extraction_data):
        data = {**valid_extraction_data, "vendor_gstin": "27AABCU9603R1"}  # 13 chars
        with pytest.raises(ValidationError):
            InvoiceExtraction(**data)

    def test_confidence_clamped_to_0_1(self, valid_extraction_data):
        data = {
            **valid_extraction_data,
            "field_confidence": {
                "vendor_gstin": 1.5,   # > 1, should clamp to 1.0
                "invoice_number": -0.1, # < 0, should clamp to 0.0
                "grand_total": 0.9,
            },
        }
        inv = InvoiceExtraction(**data)
        assert inv.field_confidence["vendor_gstin"] == 1.0
        assert inv.field_confidence["invoice_number"] == 0.0
        assert inv.field_confidence["grand_total"] == 0.9

    def test_is_inter_state_inferred_from_igst(self, valid_extraction_data):
        """If IGST > 0 and is_inter_state not set, it should be inferred as True."""
        data = {
            **valid_extraction_data,
            "tax_breakup": {
                "taxable_subtotal": 10000.00,
                "igst_amount": 1800.00,
                "grand_total": 11800.00,
            },
        }
        inv = InvoiceExtraction(**data)
        assert inv.is_inter_state is True

    def test_is_inter_state_inferred_from_cgst_sgst(self, valid_extraction_data):
        """If CGST/SGST > 0 and is_inter_state not set, it should be inferred as False."""
        inv = InvoiceExtraction(**valid_extraction_data)
        assert inv.is_inter_state is False  # CGST+SGST = intra-state

    def test_missing_vendor_name_raises(self, valid_extraction_data):
        data = {k: v for k, v in valid_extraction_data.items() if k != "vendor_name"}
        with pytest.raises(ValidationError):
            InvoiceExtraction(**data)

    def test_missing_invoice_number_raises(self, valid_extraction_data):
        data = {k: v for k, v in valid_extraction_data.items() if k != "invoice_number"}
        with pytest.raises(ValidationError):
            InvoiceExtraction(**data)

    def test_optional_buyer_fields_none(self, valid_extraction_data):
        inv = InvoiceExtraction(**valid_extraction_data)
        assert inv.buyer_name is None
        assert inv.buyer_gstin is None

    def test_date_parsing_from_string(self, valid_extraction_data):
        inv = InvoiceExtraction(**valid_extraction_data)
        assert isinstance(inv.invoice_date, date)

    def test_empty_line_items_allowed(self, valid_extraction_data):
        """Empty line_items is structurally valid; Phase 2 validator flags it."""
        data = {**valid_extraction_data, "line_items": []}
        inv = InvoiceExtraction(**data)
        assert inv.line_items == []

    def test_model_dump_round_trip(self, valid_extraction_data):
        inv = InvoiceExtraction(**valid_extraction_data)
        dumped = inv.model_dump(mode="json")
        inv2 = InvoiceExtraction.model_validate(dumped)
        assert inv2.invoice_number == inv.invoice_number
        assert inv2.tax_breakup.grand_total == inv.tax_breakup.grand_total


# ──────────────────────────────────────────────────────────────────────────────
# InvoiceRecord tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInvoiceRecord:
    def test_file_hash_computation(self):
        data = b"Hello, invoice!"
        h = InvoiceRecord.compute_file_hash(data)
        assert len(h) == 64  # SHA-256 hex = 64 chars

    def test_default_status(self, valid_extraction_data):
        inv = InvoiceExtraction(**valid_extraction_data)
        record = InvoiceRecord(
            document_id=str(uuid.uuid4()),
            file_hash="abc123",
            original_filename="test.pdf",
            raw_file_path="/storage/raw/test.pdf",
            extraction=inv,
        )
        assert record.status == "PROCESSING"
        assert record.is_duplicate is False
        assert record.needs_human_review is False
        assert isinstance(record.created_at, datetime)

    def test_record_with_errors(self):
        record = InvoiceRecord(
            document_id=str(uuid.uuid4()),
            file_hash="def456",
            original_filename="bad.pdf",
            raw_file_path="/storage/raw/bad.pdf",
            status="FAILED",
            validation_errors=["Preprocessing error: corrupt PDF"],
        )
        assert record.status == "FAILED"
        assert len(record.validation_errors) == 1
