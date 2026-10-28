"""
tests/phase1/test_preprocessing.py
=====================================
Unit tests for app/preprocessing/image_processor.py.

These tests use a synthetic image (no real invoice needed) to verify
the preprocessing pipeline works end-to-end.

Run with:
    pytest tests/phase1/test_preprocessing.py -v
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.preprocessing.image_processor import (
    PreprocessingError,
    _adaptive_binarise,
    _deskew,
    _enhance_contrast,
    _process_single_image,
    preprocess_document,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_blank_bgr(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a white BGR image with some black text-like rectangles."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Simulate "text lines"
    for y in [100, 200, 300, 400]:
        cv2.rectangle(img, (50, y), (750, y + 8), (10, 10, 10), -1)
    return img


@pytest.fixture
def blank_bgr():
    return _make_blank_bgr()


@pytest.fixture
def tmp_jpeg(tmp_path):
    """Write a synthetic invoice JPEG and return its path."""
    img = _make_blank_bgr()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    p = tmp_path / "test_invoice.jpg"
    pil.save(str(p))
    return p


@pytest.fixture
def tmp_png(tmp_path):
    img = _make_blank_bgr()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    p = tmp_path / "test_invoice.png"
    pil.save(str(p))
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for individual pipeline steps
# ──────────────────────────────────────────────────────────────────────────────

class TestDeskew:
    def test_straight_image_unchanged_shape(self, blank_bgr):
        result = _deskew(blank_bgr)
        assert result.shape == blank_bgr.shape

    def test_deskew_returns_ndarray(self, blank_bgr):
        result = _deskew(blank_bgr)
        assert isinstance(result, np.ndarray)

    def test_no_lines_returns_original(self):
        """Completely blank image (no edges) → skip deskew."""
        blank = np.ones((400, 600, 3), dtype=np.uint8) * 255
        result = _deskew(blank)
        assert result.shape == blank.shape


class TestEnhanceContrast:
    def test_output_same_shape(self, blank_bgr):
        result = _enhance_contrast(blank_bgr)
        assert result.shape == blank_bgr.shape

    def test_output_is_uint8(self, blank_bgr):
        result = _enhance_contrast(blank_bgr)
        assert result.dtype == np.uint8


class TestAdaptiveBinarise:
    def test_output_shape(self, blank_bgr):
        result = _adaptive_binarise(blank_bgr)
        assert result.shape == blank_bgr.shape

    def test_output_is_binary_values(self, blank_bgr):
        result = _adaptive_binarise(blank_bgr)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        unique = set(np.unique(gray).tolist())
        # After adaptive threshold: only 0 and 255
        assert unique.issubset({0, 255})


class TestProcessSingleImage:
    def test_full_pipeline_shape_preserved(self, blank_bgr):
        result = _process_single_image(blank_bgr)
        assert result.shape == blank_bgr.shape

    def test_returns_ndarray(self, blank_bgr):
        result = _process_single_image(blank_bgr)
        assert isinstance(result, np.ndarray)


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests for preprocess_document
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessDocument:
    def test_jpeg_input(self, tmp_jpeg, tmp_path, monkeypatch):
        """JPEG input → 1 processed page, valid base64 PNG output."""
        # Redirect output to tmp_path
        import app.preprocessing.image_processor as mod
        settings_mock = type("S", (), {
            "processed_images_dir": str(tmp_path),
        })()
        monkeypatch.setattr(mod, "get_settings", lambda: settings_mock)

        result = preprocess_document(tmp_jpeg, document_id="test-doc-001")
        assert result.page_count == 1
        assert len(result.page_base64_images) == 1
        assert len(result.saved_paths) == 1

        # Verify base64 is valid PNG
        raw = base64.b64decode(result.page_base64_images[0])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_png_input(self, tmp_png, tmp_path, monkeypatch):
        import app.preprocessing.image_processor as mod
        settings_mock = type("S", (), {"processed_images_dir": str(tmp_path)})()
        monkeypatch.setattr(mod, "get_settings", lambda: settings_mock)

        result = preprocess_document(tmp_png, document_id="test-doc-002")
        assert result.page_count == 1
        assert Path(result.saved_paths[0]).exists()

    def test_unsupported_extension_raises(self, tmp_path, monkeypatch):
        import app.preprocessing.image_processor as mod
        settings_mock = type("S", (), {"processed_images_dir": str(tmp_path)})()
        monkeypatch.setattr(mod, "get_settings", lambda: settings_mock)

        fake = tmp_path / "invoice.xyz"
        fake.write_bytes(b"not an image")
        with pytest.raises(PreprocessingError, match="Unsupported file type"):
            preprocess_document(fake)

    def test_auto_generates_document_id(self, tmp_jpeg, tmp_path, monkeypatch):
        import app.preprocessing.image_processor as mod
        settings_mock = type("S", (), {"processed_images_dir": str(tmp_path)})()
        monkeypatch.setattr(mod, "get_settings", lambda: settings_mock)

        result = preprocess_document(tmp_jpeg)  # no doc_id
        assert result.document_id  # non-empty
        assert len(result.document_id) == 36  # UUID4

    def test_saved_file_is_readable_png(self, tmp_jpeg, tmp_path, monkeypatch):
        import app.preprocessing.image_processor as mod
        settings_mock = type("S", (), {"processed_images_dir": str(tmp_path)})()
        monkeypatch.setattr(mod, "get_settings", lambda: settings_mock)

        result = preprocess_document(tmp_jpeg, document_id="test-doc-003")
        saved = Path(result.saved_paths[0])
        assert saved.suffix == ".png"
        img = cv2.imread(str(saved))
        assert img is not None
        assert img.shape[0] > 0
