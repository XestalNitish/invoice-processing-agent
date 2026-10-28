"""
app/nodes/export.py
===================
Production Export Engine for Indian Invoices:
1. Standard CSV export (flat representation of header + line items)
2. Tally Prime / Tally ERP 9 XML Voucher export (<VOUCHER VCHTYPE="Purchase">)
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.nodes.schemas import InvoiceExtraction, InvoiceRecord

logger = logging.getLogger(__name__)


def export_to_csv(record: InvoiceRecord, output_dir: Optional[Path] = None) -> Path:
    """
    Export invoice record to CSV format.
    Generates one row per line item with invoice header metadata.
    """
    settings = get_settings()
    out_dir = output_dir or Path(settings.exports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{record.document_id}_export.csv"
    inv: Optional[InvoiceExtraction] = record.extraction

    fieldnames = [
        "document_id",
        "invoice_number",
        "invoice_date",
        "vendor_name",
        "vendor_gstin",
        "buyer_name",
        "buyer_gstin",
        "place_of_supply",
        "item_description",
        "hsn_sac",
        "quantity",
        "unit",
        "unit_price",
        "discount_amount",
        "tax_rate_percent",
        "line_taxable_amount",
        "line_total_amount",
        "taxable_subtotal",
        "cgst_amount",
        "sgst_amount",
        "igst_amount",
        "cess_amount",
        "round_off",
        "grand_total",
        "status",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if not inv:
            writer.writerow({
                "document_id": record.document_id,
                "status": record.status,
            })
            return csv_path

        tb = inv.tax_breakup
        base_meta = {
            "document_id": record.document_id,
            "invoice_number": inv.invoice_number,
            "invoice_date": str(inv.invoice_date),
            "vendor_name": inv.vendor_name,
            "vendor_gstin": inv.vendor_gstin or "",
            "buyer_name": inv.buyer_name or "",
            "buyer_gstin": inv.buyer_gstin or "",
            "place_of_supply": inv.place_of_supply_state_code or "",
            "taxable_subtotal": tb.taxable_subtotal,
            "cgst_amount": tb.cgst_amount,
            "sgst_amount": tb.sgst_amount,
            "igst_amount": tb.igst_amount,
            "cess_amount": tb.cess_amount,
            "round_off": tb.round_off,
            "grand_total": tb.grand_total,
            "status": record.status,
        }

        if inv.line_items:
            for item in inv.line_items:
                row = {
                    **base_meta,
                    "item_description": item.description,
                    "hsn_sac": item.hsn_sac or "",
                    "quantity": item.quantity,
                    "unit": item.unit or "",
                    "unit_price": item.unit_price,
                    "discount_amount": item.discount_amount,
                    "tax_rate_percent": item.tax_rate_percent,
                    "line_taxable_amount": item.taxable_amount,
                    "line_total_amount": item.total_amount,
                }
                writer.writerow(row)
        else:
            writer.writerow({
                **base_meta,
                "item_description": "General Purchase",
                "hsn_sac": "",
                "quantity": 1,
                "unit": "Nos",
                "unit_price": tb.taxable_subtotal,
                "discount_amount": 0,
                "tax_rate_percent": 0,
                "line_taxable_amount": tb.taxable_subtotal,
                "line_total_amount": tb.grand_total,
            })

    logger.info("Exported CSV: %s", csv_path)
    return csv_path


def export_to_tally_xml(record: InvoiceRecord, output_dir: Optional[Path] = None) -> Path:
    """
    Generate Tally Prime / Tally.ERP 9 compatible XML import voucher.
    Root: <ENVELOPE> -> <HEADER> -> <BODY> -> <IMPORTDATA> -> <VOUCHER VCHTYPE="Purchase">
    """
    settings = get_settings()
    out_dir = output_dir or Path(settings.exports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_path = out_dir / f"{record.document_id}_tally.xml"
    inv: Optional[InvoiceExtraction] = record.extraction

    if not inv:
        raise ValueError("Cannot export empty extraction to Tally XML")

    tb = inv.tax_breakup
    tally_date = inv.invoice_date.strftime("%Y%m%d")

    envelope = ET.Element("ENVELOPE")
    
    # Header
    header = ET.SubElement(envelope, "HEADER")
    tally_req = ET.SubElement(header, "TALLYREQUEST")
    tally_req.text = "Import Data"

    # Body
    body = ET.SubElement(envelope, "BODY")
    import_data = ET.SubElement(body, "IMPORTDATA")
    
    req_desc = ET.SubElement(import_data, "REQUESTDESC")
    rep_name = ET.SubElement(req_desc, "REPORTNAME")
    rep_name.text = "All Masters"
    
    req_data = ET.SubElement(import_data, "REQUESTDATA")
    tally_msg = ET.SubElement(req_data, "TALLYMESSAGE", {"xmlns:UDF": "TallyUDF"})
    
    voucher = ET.SubElement(
        tally_msg,
        "VOUCHER",
        {
            "VCHTYPE": "Purchase",
            "ACTION": "Create",
            "OBJVIEW": "Invoice Voucher View",
        }
    )

    # Basic Voucher Info
    ET.SubElement(voucher, "DATE").text = tally_date
    ET.SubElement(voucher, "EFFECTIVEDATE").text = tally_date
    ET.SubElement(voucher, "VOUCHERTYPENAME").text = "Purchase"
    ET.SubElement(voucher, "REFERENCE").text = inv.invoice_number
    ET.SubElement(voucher, "VOUCHERNUMBER").text = inv.invoice_number
    ET.SubElement(voucher, "PARTYLEDGERNAME").text = inv.vendor_name
    ET.SubElement(voucher, "PARTYGSTIN").text = inv.vendor_gstin or ""
    ET.SubElement(voucher, "STATENAME").text = inv.place_of_supply_state_code or ""

    # Credit Party (Supplier)
    party_ledger = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
    ET.SubElement(party_ledger, "LEDGERNAME").text = inv.vendor_name
    ET.SubElement(party_ledger, "ISDEEMEDPOSITIVE").text = "No"  # Credit
    ET.SubElement(party_ledger, "AMOUNT").text = f"{tb.grand_total:.2f}"

    # Debit Purchase Accounts (by Line Items)
    for item in inv.line_items:
        ledger_name = item.suggested_ledger or f"Purchase @ {item.tax_rate_percent}%"
        item_ledger = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(item_ledger, "LEDGERNAME").text = ledger_name
        ET.SubElement(item_ledger, "ISDEEMEDPOSITIVE").text = "Yes"  # Debit
        ET.SubElement(item_ledger, "AMOUNT").text = f"-{item.taxable_amount:.2f}"

    # Debit Tax Ledgers
    if tb.cgst_amount > 0:
        cgst_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(cgst_entry, "LEDGERNAME").text = "Input CGST"
        ET.SubElement(cgst_entry, "ISDEEMEDPOSITIVE").text = "Yes"
        ET.SubElement(cgst_entry, "AMOUNT").text = f"-{tb.cgst_amount:.2f}"

    if tb.sgst_amount > 0:
        sgst_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(sgst_entry, "LEDGERNAME").text = "Input SGST"
        ET.SubElement(sgst_entry, "ISDEEMEDPOSITIVE").text = "Yes"
        ET.SubElement(sgst_entry, "AMOUNT").text = f"-{tb.sgst_amount:.2f}"

    if tb.igst_amount > 0:
        igst_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(igst_entry, "LEDGERNAME").text = "Input IGST"
        ET.SubElement(igst_entry, "ISDEEMEDPOSITIVE").text = "Yes"
        ET.SubElement(igst_entry, "AMOUNT").text = f"-{tb.igst_amount:.2f}"

    if tb.round_off != 0.0:
        ro_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(ro_entry, "LEDGERNAME").text = "Round Off"
        positive = "Yes" if tb.round_off < 0 else "No"
        ET.SubElement(ro_entry, "ISDEEMEDPOSITIVE").text = positive
        ET.SubElement(ro_entry, "AMOUNT").text = f"{-tb.round_off:.2f}"

    # Prettify and write XML
    tree = ET.ElementTree(envelope)
    ET.indent(tree, space="  ", level=0)
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)

    logger.info("Exported Tally XML: %s", xml_path)
    return xml_path
