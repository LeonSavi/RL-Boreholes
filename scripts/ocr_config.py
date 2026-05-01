"""
ocr_config.py
=============
Central configuration for Tesseract and Poppler binary paths on Windows.

Edit TESSERACT_CMD and POPPLER_PATH to match your local installation before
running the pipeline. All other modules import from here.

Windows binary installers:
  Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
             Install with 'nld' (Dutch) + 'eng' language packs selected.
  Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
             Extract to any directory; point POPPLER_PATH at the bin/ folder.
"""
from __future__ import annotations

from pathlib import Path

# ── Binary paths ───────────────────────────────────────────────────────────
TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH: Path = Path(r"C:\tools\poppler\Library\bin")

# ── OCR settings ───────────────────────────────────────────────────────────
# --oem 3 = default LSTM engine
# --psm 6 = assume uniform block of text (good default for borehole log pages)
TESSERACT_CONFIG: str = "--oem 3 --psm 6"
TESSERACT_LANG: str = "nld+eng"  # Dutch primary, English fallback for technical terms

# ── Rendering ──────────────────────────────────────────────────────────────
# 200-300 DPI is usually sufficient. Values above 300 produce very large images.
PDF_DPI: int = 200

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.parent
OCR_INPUT_DIR: Path = PROJECT_ROOT / "data" / "OCR-data"
OCR_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "ocr_output"
