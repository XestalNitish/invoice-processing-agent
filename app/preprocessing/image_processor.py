"""
app/preprocessing/image_processor.py
=====================================
Image / PDF preprocessing pipeline.

Steps applied per page
----------------------
1. PDF → image at 200 DPI (pymupdf), or direct image load
2. Deskew   – correct up to ±20° rotation using Hough line detection
3. Contrast enhance – adaptive histogram equalisation (CLAHE) on the L channel
4. Binarise – Sauvola-style adaptive threshold for crisp text
5. Denoise  – light Gaussian blur before binarisation
6. Returns  – base64-encoded PNG per page (ready for Vision LLM API)
             + writes processed PNGs to disk for the Streamlit review UI

All failures are surfaced as PreprocessingError — never silently swallowed.
"""

from __future__ import annotations

import base64
import logging
import math
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pymupdf  # replaces deprecated `import fitz`
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Raised when image/PDF preprocessing fails unrecoverably."""


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _deskew(img: np.ndarray, max_angle_deg: float = 20.0) -> np.ndarray:
    """
    Detect skew angle using Hough line transform and correct it.
    Only corrects angles within ±max_angle_deg to avoid flipping straight images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return img  # nothing to deskew

    angles: List[float] = []
    for line in lines:
        # OpenCV returns shape (N,1,4) or (N,4) depending on version
        coords = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle_deg:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img  # negligible skew

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskewed by %.2f°", median_angle)
    return rotated


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    Luminance channel of the LAB colour space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _adaptive_binarise(img: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale → Gaussian denoise → adaptive threshold.
    Returns a 3-channel BGR image (white background, black text).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _process_single_image(img_bgr: np.ndarray) -> np.ndarray:
    """Apply the full preprocessing pipeline to a single page image."""
    img = _deskew(img_bgr)
    img = _enhance_contrast(img)
    img = _adaptive_binarise(img)
    return img


def _encode_to_base64_png(img_bgr: np.ndarray) -> str:
    """Encode a BGR ndarray to a base64 PNG string for the Vision LLM."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise PreprocessingError("Failed to encode processed image to PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pdf_to_page_images(pdf_path: Path, dpi: int = 200) -> List[np.ndarray]:
    """Render each PDF page to an ndarray via pymupdf at the given DPI."""
    doc = pymupdf.open(str(pdf_path))
    pages: List[np.ndarray] = []
    zoom = dpi / 72.0  # 72 = PDF default DPI
    mat = pymupdf.Matrix(zoom, zoom)

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        pages.append(arr_bgr)
        logger.debug("Rendered PDF page %d (%dx%d)", page_num + 1, pix.width, pix.height)

    doc.close()
    return pages


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class ProcessedDocument:
    """Result of preprocessing a single invoice file."""

    def __init__(
        self,
        document_id: str,
        page_base64_images: List[str],
        saved_paths: List[str],
        page_count: int,
    ) -> None:
        self.document_id = document_id
        self.page_base64_images = page_base64_images  # for LLM API
        self.saved_paths = saved_paths                # for Streamlit UI / storage
        self.page_count = page_count


def preprocess_document(
    file_path: Path,
    document_id: str | None = None,
    dpi: int = 200,
) -> ProcessedDocument:
    """
    Main entry point.

    Parameters
    ----------
    file_path : Path
        Path to a .pdf, .jpg, .jpeg, .png, or .tiff file.
    document_id : str, optional
        Identifier used for naming saved output files.
        Auto-generated if not provided.
    dpi : int
        Resolution for PDF rendering (default 200).

    Returns
    -------
    ProcessedDocument
        Contains per-page base64 PNGs and saved file paths.

    Raises
    ------
    PreprocessingError
        On unsupported file type or processing failure.
    """
    settings = get_settings()
    doc_id = document_id or str(uuid.uuid4())
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # ── Load pages ──────────────────────────────────────────────────────
    try:
        if suffix == ".pdf":
            raw_pages = _pdf_to_page_images(file_path, dpi=dpi)
        elif suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}:
            pil_img = Image.open(file_path)
            # Handle multi-page TIFFs
            frames: List[np.ndarray] = []
            try:
                for frame_idx in range(getattr(pil_img, "n_frames", 1)):
                    pil_img.seek(frame_idx)
                    frames.append(_pil_to_cv2(pil_img.copy()))
            except EOFError:
                pass
            raw_pages = frames if frames else [_pil_to_cv2(pil_img)]
        else:
            raise PreprocessingError(
                f"Unsupported file type '{suffix}'. "
                "Supported: .pdf, .jpg, .jpeg, .png, .tiff, .bmp, .webp"
            )
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Failed to load '{file_path.name}': {exc}") from exc

    if not raw_pages:
        raise PreprocessingError(f"No pages found in '{file_path.name}'")

    # ── Process each page ────────────────────────────────────────────────
    out_dir = Path(settings.processed_images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_base64: List[str] = []
    saved_paths: List[str] = []

    for idx, page_bgr in enumerate(raw_pages):
        try:
            processed = _process_single_image(page_bgr)
        except Exception as exc:
            raise PreprocessingError(
                f"Processing failed on page {idx + 1} of '{file_path.name}': {exc}"
            ) from exc

        # Save to disk
        out_name = f"{doc_id}_page{idx + 1:02d}.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), processed)
        saved_paths.append(str(out_path))

        # Encode for LLM
        page_base64.append(_encode_to_base64_png(processed))
        logger.info("Preprocessed page %d/%d → %s", idx + 1, len(raw_pages), out_path.name)

    return ProcessedDocument(
        document_id=doc_id,
        page_base64_images=page_base64,
        saved_paths=saved_paths,
        page_count=len(raw_pages),
    )
