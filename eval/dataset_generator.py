"""
eval/dataset_generator.py
=========================
Generates a realistic benchmark dataset of Indian GST invoices (both text and images)
with exact Ground Truth labels for automated model evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5 Diverse Ground Truth Indian Invoices
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_CASES = [
    {
        "id": "INV_001_MAH_INTRASTATE",
        "description": "Maharashtra Intra-State B2B Electronics (CGST + SGST)",
        "ground_truth": {
            "vendor_name": "TECHFLOW SOLUTIONS PVT LTD",
            "vendor_address": "Plot 42, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra - 400093",
            "vendor_gstin": "27AABCT8842K1ZU",
            "vendor_phone": "+91 22 2834 9000",
            "buyer_name": "APEX DATA SYSTEMS LLP",
            "buyer_gstin": "27AAACA1234D1ZF",
            "buyer_address": "Office 502, Commerce Center, Pune, Maharashtra - 411004",
            "invoice_number": "TF/2024-25/0412",
            "invoice_date": "2024-04-15",
            "place_of_supply_state_code": "27",
            "is_inter_state": False,
            "line_items": [
                {
                    "description": "Samsung 980 Pro 1TB NVMe SSD",
                    "hsn_sac": "8471",
                    "quantity": 5.0,
                    "unit": "Nos",
                    "unit_price": 8500.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 18.0,
                    "taxable_amount": 42500.00,
                    "total_amount": 50150.00,
                },
                {
                    "description": "Corsair Vengeance 32GB DDR5 RAM",
                    "hsn_sac": "8473",
                    "quantity": 4.0,
                    "unit": "Nos",
                    "unit_price": 9200.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 18.0,
                    "taxable_amount": 36800.00,
                    "total_amount": 43424.00,
                }
            ],
            "tax_breakup": {
                "taxable_subtotal": 79300.00,
                "cgst_amount": 7137.00,
                "sgst_amount": 7137.00,
                "igst_amount": 0.00,
                "cess_amount": 0.00,
                "tds_amount": 0.00,
                "round_off": 0.00,
                "grand_total": 93574.00,
            }
        },
        "raw_text": """
================================================================================
                                TAX INVOICE
================================================================================
Seller: TECHFLOW SOLUTIONS PVT LTD
Address: Plot 42, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra - 400093
GSTIN: 27AABCT8842K1ZU | Phone: +91 22 2834 9000

Buyer: APEX DATA SYSTEMS LLP
Address: Office 502, Commerce Center, Pune, Maharashtra - 411004
GSTIN: 27AAACA1234D1ZF

Invoice No: TF/2024-25/0412              Invoice Date: 2024-04-15
Place of Supply: 27 (Maharashtra)         State Code: 27

--------------------------------------------------------------------------------
Item Description                  HSN    Qty   Rate        Taxable     Total (INR)
--------------------------------------------------------------------------------
1. Samsung 980 Pro 1TB NVMe SSD   8471   5 Nos 8,500.00    42,500.00   50,150.00
2. Corsair Vengeance 32GB DDR5    8473   4 Nos 9,200.00    36,800.00   43,424.00
--------------------------------------------------------------------------------
Subtotal (Taxable Value):                                    INR 79,300.00
Central GST (CGST @ 9%):                                     INR  7,137.00
State GST (SGST @ 9%):                                       INR  7,137.00
Integrated GST (IGST @ 0%):                                  INR      0.00
Round Off:                                                   INR      0.00
GRAND TOTAL:                                                 INR 93,574.00
================================================================================
"""
    },
    {
        "id": "INV_002_DEL_INTERSTATE",
        "description": "Delhi to Karnataka Inter-State Supply (IGST)",
        "ground_truth": {
            "vendor_name": "BHARAT INDUSTRIAL SPARES",
            "vendor_address": "88 Industrial Estate, Okhla Phase 3, New Delhi - 110020",
            "vendor_gstin": "07AABCB4412M1Z8",
            "vendor_phone": "+91 11 2681 4400",
            "buyer_name": "DECCAN PRECISION ENGINEERING",
            "buyer_gstin": "29AABCD9876P1Z3",
            "buyer_address": "Peenya Industrial Area, Bengaluru, Karnataka - 560058",
            "invoice_number": "BIS/24/0988",
            "invoice_date": "2024-05-20",
            "place_of_supply_state_code": "29",
            "is_inter_state": True,
            "line_items": [
                {
                    "description": "Heavy Duty Industrial Bearings 6205",
                    "hsn_sac": "8482",
                    "quantity": 50.0,
                    "unit": "Pcs",
                    "unit_price": 340.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 18.0,
                    "taxable_amount": 17000.00,
                    "total_amount": 20060.00,
                }
            ],
            "tax_breakup": {
                "taxable_subtotal": 17000.00,
                "cgst_amount": 0.00,
                "sgst_amount": 0.00,
                "igst_amount": 3060.00,
                "cess_amount": 0.00,
                "tds_amount": 0.00,
                "round_off": 0.00,
                "grand_total": 20060.00,
            }
        },
        "raw_text": """
TAX INVOICE (INTER-STATE)
Vendor: BHARAT INDUSTRIAL SPARES
Address: 88 Industrial Estate, Okhla Phase 3, New Delhi - 110020
GSTIN: 07AABCB4412M1Z8 | Phone: +91 11 2681 4400

Bill To: DECCAN PRECISION ENGINEERING
Address: Peenya Industrial Area, Bengaluru, Karnataka - 560058
GSTIN: 29AABCD9876P1Z3

Invoice Number: BIS/24/0988
Invoice Date: 2024-05-20
Place of Supply: 29 (Karnataka)

Line Items:
1. Heavy Duty Industrial Bearings 6205 | HSN: 8482 | Qty: 50 Pcs | Rate: 340.00 | Taxable: 17000.00 | IGST: 18% | Total: 20060.00

Tax Summary:
Taxable Subtotal: INR 17,000.00
CGST: 0.00
SGST: 0.00
IGST @ 18%: 3,060.00
Grand Total: INR 20,060.00
"""
    },
    {
        "id": "INV_003_SERVICE_CONSULTING",
        "description": "Professional IT Services Retainer (SAC 9983, 18% GST)",
        "ground_truth": {
            "vendor_name": "CLOUDMIND TECHNOLOGIES",
            "vendor_address": "Level 4, Cyber City, Gurugram, Haryana - 122002",
            "vendor_gstin": "06AABCC9911D1ZX",
            "vendor_phone": "+91 124 456 7890",
            "buyer_name": "STARLIGHT MEDIA WORKS",
            "buyer_gstin": "06AAACS5544B1ZV",
            "buyer_address": "Sector 18, Gurugram, Haryana",
            "invoice_number": "CMT-SRV-2024-11",
            "invoice_date": "2024-06-01",
            "place_of_supply_state_code": "06",
            "is_inter_state": False,
            "line_items": [
                {
                    "description": "Cloud Architecture & DevOps Consulting Retainer (May 2024)",
                    "hsn_sac": "9983",
                    "quantity": 1.0,
                    "unit": "Month",
                    "unit_price": 60000.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 18.0,
                    "taxable_amount": 60000.00,
                    "total_amount": 70800.00,
                }
            ],
            "tax_breakup": {
                "taxable_subtotal": 60000.00,
                "cgst_amount": 5400.00,
                "sgst_amount": 5400.00,
                "igst_amount": 0.00,
                "cess_amount": 0.00,
                "tds_amount": 0.00,
                "round_off": 0.00,
                "grand_total": 70800.00,
            }
        },
        "raw_text": """
TAX INVOICE FOR SERVICES
Service Provider: CLOUDMIND TECHNOLOGIES
Address: Level 4, Cyber City, Gurugram, Haryana - 122002
GSTIN: 06AABCC9911D1ZX | Tel: +91 124 456 7890

Client: STARLIGHT MEDIA WORKS
Address: Sector 18, Gurugram, Haryana
GSTIN: 06AAACS5544B1ZV

Invoice No: CMT-SRV-2024-11
Date: 2024-06-01
Place of Supply: 06 (Haryana)

Description of Service:
1. Cloud Architecture & DevOps Consulting Retainer (May 2024)
   SAC Code: 9983 | Qty: 1 Month | Rate: 60,000.00 | Taxable: 60,000.00 | Total: 70,800.00

Tax Calculation:
Taxable Value: 60,000.00
CGST 9%: 5,400.00
SGST 9%: 5,400.00
IGST: 0.00
Total Amount Payable: INR 70,800.00
"""
    },
    {
        "id": "INV_004_KIRANA_UNREGISTERED",
        "description": "Small Kirana Store Bill (No GSTIN, Small Vendor)",
        "ground_truth": {
            "vendor_name": "GUPTA GENERAL STORE",
            "vendor_address": "Shop No 3, Main Market, Laxmi Nagar, Delhi - 110092",
            "vendor_gstin": None,
            "vendor_phone": "+91 98112 34567",
            "buyer_name": "SUNIL KUMAR",
            "buyer_gstin": None,
            "buyer_address": "Laxmi Nagar, Delhi",
            "invoice_number": "GGS-942",
            "invoice_date": "2024-06-10",
            "place_of_supply_state_code": "07",
            "is_inter_state": False,
            "line_items": [
                {
                    "description": "Basmati Rice 5kg Bag",
                    "hsn_sac": "1006",
                    "quantity": 2.0,
                    "unit": "Bags",
                    "unit_price": 550.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 0.0,
                    "taxable_amount": 1100.00,
                    "total_amount": 1100.00,
                },
                {
                    "description": "Sunflower Cooking Oil 1L",
                    "hsn_sac": "1512",
                    "quantity": 3.0,
                    "unit": "Bottles",
                    "unit_price": 160.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 0.0,
                    "taxable_amount": 480.00,
                    "total_amount": 480.00,
                }
            ],
            "tax_breakup": {
                "taxable_subtotal": 1580.00,
                "cgst_amount": 0.00,
                "sgst_amount": 0.00,
                "igst_amount": 0.00,
                "cess_amount": 0.00,
                "tds_amount": 0.00,
                "round_off": 0.00,
                "grand_total": 1580.00,
            }
        },
        "raw_text": """
RETAIL BILL / CASH MEMO
GUPTA GENERAL STORE
Shop No 3, Main Market, Laxmi Nagar, Delhi - 110092
Phone: +91 98112 34567 (Composition / Unregistered Dealer)

Customer: SUNIL KUMAR, Laxmi Nagar, Delhi
Bill No: GGS-942
Bill Date: 2024-06-10

1. Basmati Rice 5kg Bag - Qty: 2 Bags @ 550.00 = 1,100.00
2. Sunflower Cooking Oil 1L - Qty: 3 Bottles @ 160.00 = 480.00

Taxable Subtotal: 1,580.00
Tax: 0.00
NET AMOUNT: 1,580.00
"""
    },
    {
        "id": "INV_005_ROUNDOFF_MULTIITEM",
        "description": "Commercial Stationery with Fractional Discount and Round Off",
        "ground_truth": {
            "vendor_name": "ROYAL STATIONERY MART",
            "vendor_address": "22 M.G. Road, Fort, Bengaluru, Karnataka - 560001",
            "vendor_gstin": "29AABCR5511A1ZZ",
            "vendor_phone": "+91 80 2221 3344",
            "buyer_name": "TECH TALENT INC",
            "buyer_gstin": "29AAACT9012K1Z9",
            "buyer_address": "Indiranagar, Bengaluru",
            "invoice_number": "RSM/2024/782",
            "invoice_date": "2024-06-25",
            "place_of_supply_state_code": "29",
            "is_inter_state": False,
            "line_items": [
                {
                    "description": "Whiteboard Markers Assorted Box of 12",
                    "hsn_sac": "9608",
                    "quantity": 10.0,
                    "unit": "Boxes",
                    "unit_price": 320.00,
                    "discount_amount": 200.00,
                    "tax_rate_percent": 18.0,
                    "taxable_amount": 3000.00,
                    "total_amount": 3540.00,
                },
                {
                    "description": "Spiral Bound Project Diaries",
                    "hsn_sac": "4820",
                    "quantity": 15.0,
                    "unit": "Pcs",
                    "unit_price": 145.00,
                    "discount_amount": 0.0,
                    "tax_rate_percent": 12.0,
                    "taxable_amount": 2175.00,
                    "total_amount": 2436.00,
                }
            ],
            "tax_breakup": {
                "taxable_subtotal": 5175.00,
                "cgst_amount": 265.50,
                "sgst_amount": 265.50,
                "igst_amount": 0.00,
                "cess_amount": 0.00,
                "tds_amount": 0.00,
                "round_off": 0.00,
                "grand_total": 5976.00,
            }
        },
        "raw_text": """
TAX INVOICE
ROYAL STATIONERY MART
Address: 22 M.G. Road, Fort, Bengaluru, Karnataka - 560001
GSTIN: 29AABCR5511A1ZZ | Tel: +91 80 2221 3344

Customer: TECH TALENT INC
Address: Indiranagar, Bengaluru | GSTIN: 29AAACT9012K1Z9

Invoice No: RSM/2024/782
Date: 2024-06-25
Place of Supply: 29 (Karnataka)

Items:
1. Whiteboard Markers Assorted Box of 12 | HSN: 9608 | Qty: 10 Boxes | Rate: 320.00 | Disc: 200.00 | Taxable: 3000.00 | GST: 18% | Total: 3540.00
2. Spiral Bound Project Diaries | HSN: 4820 | Qty: 15 Pcs | Rate: 145.00 | Disc: 0.00 | Taxable: 2175.00 | GST: 12% | Total: 2436.00

Tax Breakdown:
Taxable Value: 5,175.00
CGST (9% on markers + 6% on diaries): 265.50
SGST (9% on markers + 6% on diaries): 265.50
IGST: 0.00
Round off: 0.00
GRAND TOTAL: INR 5,976.00
"""
    }
]


def render_invoice_image(invoice_id: str, text: str, out_path: Path):
    """Render text as an invoice PNG image for OCR & vision tests."""
    lines = text.strip().splitlines()
    width = 1000
    line_height = 24
    height = max(800, len(lines) * line_height + 80)
    
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try basic font or default
    y = 30
    for line in lines:
        draw.text((40, y), line, fill=(20, 20, 20))
        y += line_height
        
    img.save(str(out_path))


def generate_dataset():
    """Create all benchmark files and ground truth JSON."""
    gt_map = {}
    print(f"Generating benchmark dataset in {DATASET_DIR}...")
    
    for case in BENCHMARK_CASES:
        cid = case["id"]
        # Save raw text
        txt_path = DATASET_DIR / f"{cid}.txt"
        txt_path.write_text(case["raw_text"].strip(), encoding="utf-8")
        
        # Save synthetic rendered image
        img_path = DATASET_DIR / f"{cid}.png"
        render_invoice_image(cid, case["raw_text"], img_path)
        
        gt_map[cid] = {
            "description": case["description"],
            "txt_path": str(txt_path),
            "img_path": str(img_path),
            "ground_truth": case["ground_truth"]
        }
        print(f"  ✅ {cid} ({case['description']})")

    gt_file = DATASET_DIR / "ground_truth.json"
    gt_file.write_text(json.dumps(gt_map, indent=2), encoding="utf-8")
    print(f"🎉 Benchmark dataset generated: {len(BENCHMARK_CASES)} invoices saved to {DATASET_DIR}")
    return gt_file


if __name__ == "__main__":
    generate_dataset()
