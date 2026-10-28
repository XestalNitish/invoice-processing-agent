"""
ui/review_app.py
================
Production Streamlit Human-in-the-Loop Review Dashboard.
- Left column: Preprocessed invoice image viewer (with multi-page tabs)
- Right column: Interactive form with low-confidence field highlights
- Real-time deterministic validation and arithmetic check
- Approve / Reject buttons with automatic correction logging
- One-click CSV and Tally-compatible XML voucher export
"""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path configuration
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests
import streamlit as st
from PIL import Image

from app.core.config import get_settings
from app.db.crud import get_by_document_id, list_needs_review, update_invoice_record, log_correction
from app.db.session import get_session
from app.nodes.export import export_to_csv, export_to_tally_xml
from app.nodes.schemas import InvoiceExtraction, InvoiceRecord, LineItem, TaxBreakup
from app.nodes.validators import validate_invoice
from app.pipeline_phase1 import run_phase1

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration & Styling
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GST Invoice Review Portal",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 8px;
    }
    .badge-approved { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .badge-review { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-failed { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    
    .low-conf-field {
        border-left: 4px solid #f59e0b !important;
        background-color: #fffbeb !important;
        padding: 6px 10px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .conf-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #d97706;
        font-weight: bold;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_records_from_db(status_filter: Optional[str] = None) -> List[InvoiceRecord]:
    """Retrieve invoice records directly from the database."""
    try:
        with get_session() as s:
            from app.db.models import InvoiceRecordORM
            q = s.query(InvoiceRecordORM)
            if status_filter:
                q = q.filter(InvoiceRecordORM.status == status_filter)
            q = q.order_by(InvoiceRecordORM.created_at.desc()).limit(100)
            from app.db.crud import _orm_to_record
            return [_orm_to_record(o) for o in q.all()]
    except Exception as e:
        st.sidebar.warning(f"DB load fallback: {e}")
        return []


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Navigation & Document Queue
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🧾 Invoice Agent")
api_online = check_api_health()
st.sidebar.caption(
    f"Backend API: {'🟢 Online' if api_online else '🟡 Standalone DB Mode'} • 100% On-Premise"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Upload New Invoice")
uploaded_file = st.sidebar.file_uploader(
    "Choose PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "webp"],
    key="uploader_input",
)

if uploaded_file is not None and st.sidebar.button("⚡ Process Invoice", use_container_width=True):
    with st.spinner("Processing invoice (Preprocess → OCR/LLM → Validation)..."):
        settings = get_settings()
        temp_dir = Path(settings.raw_images_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        save_path = temp_dir / uploaded_file.name
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            record = run_phase1(file_path=save_path, persist_to_db=True)
            if record.extraction:
                v_res = validate_invoice(record.extraction)
                record.validation_errors = v_res.errors
                record.low_confidence_fields = v_res.low_confidence_fields
                record.needs_human_review = v_res.needs_review
                record.status = "APPROVED" if v_res.is_valid and not v_res.needs_review else "NEEDS_REVIEW"
                with get_session() as s:
                    update_invoice_record(s, record)

            st.sidebar.success(f"Processed: {record.document_id[:8]}... [{record.status}]")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Processing failed: {exc}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Document Queue")
status_filter = st.sidebar.selectbox(
    "Filter by Status",
    ["ALL", "NEEDS_REVIEW", "APPROVED", "REJECTED"],
    index=1,
)
filter_val = None if status_filter == "ALL" else status_filter
records = fetch_records_from_db(status_filter=filter_val)

if not records:
    st.sidebar.info("No invoices found in this queue.")
    selected_record = None
else:
    doc_labels = {
        r.document_id: f"{r.original_filename[:20]} | {r.status} ({r.created_at.strftime('%H:%M:%S')})"
        for r in records
    }
    selected_id = st.sidebar.radio(
        "Select Document",
        options=list(doc_labels.keys()),
        format_func=lambda x: doc_labels[x],
        key="selected_doc_radio",
    )
    selected_record = next((r for r in records if r.document_id == selected_id), None)


# ──────────────────────────────────────────────────────────────────────────────
# Main Screen: Split-Screen Review Portal
# ──────────────────────────────────────────────────────────────────────────────

if selected_record is None:
    st.title("GST Invoice Human Review & Export Portal")
    st.info("👈 Upload an invoice from the sidebar or select a document from the queue to start reviewing.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>100% Private</h3><p>Runs entirely offline on CPU/Postgres</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>Zero Hallucinations</h3><p>Deterministic Indian GST & Modulo-36 Checksum</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Tally XML Ready</h3><p>Instant Export for Tally Prime & ERP 9</p></div>', unsafe_allow_html=True)

else:
    record = selected_record
    inv = record.extraction

    # Top Status Bar
    header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
    with header_col1:
        st.subheader(f"Document: {record.original_filename}")
        badge_class = "badge-approved" if record.status == "APPROVED" else ("badge-review" if record.status == "NEEDS_REVIEW" else "badge-failed")
        st.markdown(f'<span class="status-badge {badge_class}">Status: {record.status}</span>', unsafe_allow_html=True)
    with header_col2:
        st.caption(f"Doc ID: `{record.document_id[:16]}...`")
        st.caption(f"File Hash: `{record.file_hash[:12]}...`")
    with header_col3:
        if record.is_duplicate:
            st.error(f"⚠️ Duplicate of: `{record.duplicate_of}`")

    # Display Validation Alerts
    if record.validation_errors:
        st.warning("⚠️ **Deterministic Validation Issues Detected:**\n" + "\n".join([f"- {e}" for e in record.validation_errors]))

    st.markdown("---")

    # Split Screen: Left (Image) vs Right (Editable Form)
    left_col, right_col = st.columns([1, 1], gap="large")

    # ─────────────────────────────────────────────────────────
    # LEFT COLUMN: Invoice Image Viewer
    # ─────────────────────────────────────────────────────────
    with left_col:
        st.markdown("### 🖼️ Original Document Preview")
        image_paths = record.processed_image_paths or []
        
        # Fallback to raw file if no processed images
        if not image_paths and record.raw_file_path and Path(record.raw_file_path).suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image_paths = [record.raw_file_path]

        if image_paths:
            if len(image_paths) > 1:
                tabs = st.tabs([f"Page {i+1}" for i in range(len(image_paths))])
                for i, tab in enumerate(tabs):
                    with tab:
                        img_p = Path(image_paths[i])
                        if img_p.exists():
                            st.image(str(img_p), use_container_width=True)
            else:
                img_p = Path(image_paths[0])
                if img_p.exists():
                    st.image(str(img_p), use_container_width=True)
                else:
                    st.warning("Image file not found on disk.")
        else:
            st.info("Document is stored as PDF without rendered preview. Raw path: " + record.raw_file_path)

    # ─────────────────────────────────────────────────────────
    # RIGHT COLUMN: Structured & Editable Data
    # ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown("### ✍️ Extracted & Editable Fields")

        if not inv:
            st.error("No extraction data available for this document.")
        else:
            conf = inv.field_confidence or {}
            low_conf = record.low_confidence_fields or []

            with st.form("invoice_review_form"):
                # 1. Vendor Details
                st.markdown("##### 🏢 Vendor Information")
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    vendor_name = st.text_input("Vendor / Seller Name", value=inv.vendor_name)
                with v_col2:
                    is_low_gstin = "vendor_gstin" in low_conf or conf.get("vendor_gstin", 1.0) < 0.85
                    gstin_label = f"Vendor GSTIN {'⚠️ (Low Confidence)' if is_low_gstin else ''}"
                    vendor_gstin = st.text_input(gstin_label, value=inv.vendor_gstin or "")
                
                vendor_address = st.text_area("Vendor Address", value=inv.vendor_address or "", height=68)

                # 2. Buyer & Invoice Details
                st.markdown("##### 📄 Invoice Identity")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    inv_no = st.text_input("Invoice Number", value=inv.invoice_number)
                with m_col2:
                    inv_date = st.date_input("Invoice Date", value=inv.invoice_date)
                with m_col3:
                    pos = st.text_input("Place of Supply (State Code)", value=inv.place_of_supply_state_code or "")

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    buyer_name = st.text_input("Buyer / Customer Name", value=inv.buyer_name or "")
                with b_col2:
                    buyer_gstin = st.text_input("Buyer GSTIN", value=inv.buyer_gstin or "")

                # 3. Line Items
                st.markdown("##### 📦 Line Items")
                items_data = []
                for idx, item in enumerate(inv.line_items):
                    st.caption(f"Item #{idx+1}")
                    ic1, ic2, ic3, ic4 = st.columns([3, 1, 1, 1])
                    with ic1:
                        desc = st.text_input(f"Desc #{idx+1}", value=item.description, label_visibility="collapsed")
                    with ic2:
                        qty = st.number_input(f"Qty #{idx+1}", value=float(item.quantity), min_value=0.0, step=1.0)
                    with ic3:
                        rate = st.number_input(f"Rate #{idx+1}", value=float(item.unit_price), min_value=0.0, step=10.0)
                    with ic4:
                        tot = st.number_input(f"Total #{idx+1}", value=float(item.total_amount), min_value=0.0, step=10.0)
                    
                    items_data.append({
                        "description": desc,
                        "hsn_sac": item.hsn_sac,
                        "quantity": qty,
                        "unit": item.unit,
                        "unit_price": rate,
                        "discount_amount": item.discount_amount,
                        "tax_rate_percent": item.tax_rate_percent,
                        "taxable_amount": round(qty * rate, 2),
                        "total_amount": tot,
                        "suggested_ledger": item.suggested_ledger,
                    })

                # 4. Tax Breakup
                st.markdown("##### 💰 Tax & Grand Total")
                tb = inv.tax_breakup
                t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                with t_col1:
                    taxable_subtotal = st.number_input("Taxable Subtotal", value=float(tb.taxable_subtotal), step=100.0)
                with t_col2:
                    cgst_amt = st.number_input("CGST (₹)", value=float(tb.cgst_amount), step=10.0)
                with t_col3:
                    sgst_amt = st.number_input("SGST (₹)", value=float(tb.sgst_amount), step=10.0)
                with t_col4:
                    igst_amt = st.number_input("IGST (₹)", value=float(tb.igst_amount), step=10.0)

                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    round_off = st.number_input("Round Off (₹)", value=float(tb.round_off), step=0.1)
                with g_col2:
                    grand_total = st.number_input("Grand Total (₹)", value=float(tb.grand_total), step=100.0)

                reviewer_notes = st.text_input("Reviewer Notes (Optional)", value=record.reviewer_notes or "")

                # Action Buttons inside Form
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    approve_submitted = st.form_submit_button("✅ Approve Invoice", use_container_width=True)
                with btn_col2:
                    reject_submitted = st.form_submit_button("❌ Reject Invoice", use_container_width=True)

            # Handling Form Submission
            if approve_submitted or reject_submitted:
                decision = "APPROVED" if approve_submitted else "REJECTED"
                
                # Check for human edits to log
                corrected_fields = {}
                if vendor_name != inv.vendor_name: corrected_fields["vendor_name"] = vendor_name
                if (vendor_gstin or None) != inv.vendor_gstin: corrected_fields["vendor_gstin"] = vendor_gstin or None
                if inv_no != inv.invoice_number: corrected_fields["invoice_number"] = inv_no
                if inv_date != inv.invoice_date: corrected_fields["invoice_date"] = str(inv_date)
                if grand_total != tb.grand_total: corrected_fields["grand_total"] = grand_total

                # Update Extraction Object
                updated_tax = TaxBreakup(
                    taxable_subtotal=taxable_subtotal,
                    cgst_amount=cgst_amt,
                    sgst_amount=sgst_amt,
                    igst_amount=igst_amt,
                    cess_amount=tb.cess_amount,
                    tds_amount=tb.tds_amount,
                    round_off=round_off,
                    grand_total=grand_total,
                )
                updated_extraction = InvoiceExtraction(
                    vendor_name=vendor_name,
                    vendor_address=vendor_address or None,
                    vendor_gstin=vendor_gstin.strip().upper() if vendor_gstin else None,
                    vendor_phone=inv.vendor_phone,
                    buyer_name=buyer_name or None,
                    buyer_gstin=buyer_gstin.strip().upper() if buyer_gstin else None,
                    buyer_address=inv.buyer_address,
                    invoice_number=inv_no,
                    invoice_date=inv_date,
                    place_of_supply_state_code=pos or None,
                    is_inter_state=(igst_amt > 0),
                    line_items=[LineItem(**it) for it in items_data],
                    tax_breakup=updated_tax,
                    field_confidence=conf,
                )

                # Persist to Database & Log corrections
                with get_session() as session:
                    for k, val in corrected_fields.items():
                        log_correction(
                            session=session,
                            document_id=record.document_id,
                            field_name=k,
                            original_value=str(getattr(inv, k, "")),
                            corrected_value=str(val),
                            reviewed_by="human_ui_reviewer",
                        )
                    
                    record.extraction = updated_extraction
                    record.status = decision
                    record.human_reviewed = True
                    record.reviewed_by = "human_ui_reviewer"
                    record.reviewed_at = datetime.utcnow()
                    record.reviewer_notes = reviewer_notes
                    record.needs_human_review = False
                    
                    # Generate Exports on Approval
                    if decision == "APPROVED":
                        csv_p = export_to_csv(record)
                        xml_p = export_to_tally_xml(record)
                        record.export_csv_path = str(csv_p)
                        record.export_tally_xml_path = str(xml_p)

                    update_invoice_record(session, record)

                st.success(f"Invoice marked as {decision}! Labelled corrections saved.")
                st.rerun()

            # Export Download Section
            st.markdown("---")
            st.markdown("##### 📥 Export Records")
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                csv_file = export_to_csv(record)
                with open(csv_file, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📊 Download CSV",
                        data=f.read(),
                        file_name=f"{record.document_id[:8]}_export.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with exp_col2:
                xml_file = export_to_tally_xml(record)
                with open(xml_file, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="🏦 Download Tally XML",
                        data=f.read(),
                        file_name=f"{record.document_id[:8]}_tally.xml",
                        mime="application/xml",
                        use_container_width=True,
                    )
