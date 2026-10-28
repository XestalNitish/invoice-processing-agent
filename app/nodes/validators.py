import re
from typing import List, Dict, Any, Optional
from .schemas import InvoiceExtraction, LineItem, TaxBreakup
from app.graph.state import InvoiceAgentState

GSTIN_REGEX = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def verify_gstin_checksum(gstin: str) -> bool:
    """Official Indian GSTN Modulo-36 Checksum Validator."""
    if not gstin or not re.match(GSTIN_REGEX, gstin):
        return False
    total = 0
    for i, char in enumerate(gstin[:14]):
        val = CHARSET.index(char)
        factor = 2 if (i % 2 != 0) else 1
        prod = val * factor
        total += (prod // 36) + (prod % 36)
    check_code = (36 - (total % 36)) % 36
    return gstin[14] == CHARSET[check_code]


def validate_node(state: InvoiceAgentState) -> dict:
    inv = state["extracted_data"]
    errors = []
    low_conf = []

    # 1. GSTIN Format & Checksum
    if inv.vendor_gstin:
        if not verify_gstin_checksum(inv.vendor_gstin.upper()):
            errors.append(f"Invalid Vendor GSTIN '{inv.vendor_gstin}' (failed regex or Modulo-36 checksum).")
            low_conf.append("vendor_gstin")
    else:
        low_conf.append("vendor_gstin")

    # 2. Line Items Sum vs Taxable Subtotal
    calc_subtotal = round(sum(item.taxable_amount for item in inv.line_items), 2)
    if abs(calc_subtotal - inv.tax_breakup.taxable_subtotal) > 1.0:
        errors.append(
            f"Line items sum ({calc_subtotal}) does not match taxable_subtotal ({inv.tax_breakup.taxable_subtotal})."
        )
        low_conf.append("line_items")

    # 3. CGST/SGST Integrity (Intra-state CGST should equal SGST)
    tb = inv.tax_breakup
    if (tb.cgst_amount > 0 or tb.sgst_amount > 0) and abs(tb.cgst_amount - tb.sgst_amount) > 0.5:
        errors.append(f"CGST ({tb.cgst_amount}) and SGST ({tb.sgst_amount}) mismatch.")
        low_conf.append("tax_breakup")

    # 4. Grand Total Arithmetic Check
    expected_grand = round(
        tb.taxable_subtotal + tb.cgst_amount + tb.sgst_amount + tb.igst_amount + tb.cess_amount + tb.round_off, 2
    )
    if abs(expected_grand - tb.grand_total) > 1.0:
        errors.append(
            f"Computed grand total ({expected_grand}) != extracted grand_total ({tb.grand_total})."
        )
        low_conf.append("grand_total")

    # 5. Low Confidence Threshold (< 0.85)
    for field_name, score in inv.field_confidence.items():
        if score < 0.85 and field_name not in low_conf:
            low_conf.append(field_name)

    is_valid = len(errors) == 0
    needs_review = (not is_valid) or (len(low_conf) > 0) or state.get("is_duplicate", False)

    return {
        "is_valid": is_valid,
        "validation_errors": errors,
        "low_confidence_fields": low_conf,
        "needs_human_review": needs_review,
    }


class ValidationResult:
    def __init__(self, is_valid: bool, errors: List[str], low_confidence_fields: List[str], needs_review: bool):
        self.is_valid = is_valid
        self.errors = errors
        self.low_confidence_fields = low_confidence_fields
        self.needs_review = needs_review

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "low_confidence_fields": self.low_confidence_fields,
            "needs_review": self.needs_review,
        }


def validate_invoice(
    extraction: InvoiceExtraction,
    confidence_threshold: float = 0.85,
    is_duplicate: bool = False,
) -> ValidationResult:
    fake_state: Dict[str, Any] = {
        "extracted_data": extraction,
        "is_duplicate": is_duplicate,
    }
    res = validate_node(fake_state)
    return ValidationResult(
        is_valid=res["is_valid"],
        errors=res["validation_errors"],
        low_confidence_fields=res["low_confidence_fields"],
        needs_review=res["needs_human_review"],
    )
