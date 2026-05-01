"""
ocr_pipeline.py
===============
Tesseract OCR pipeline: PDF → structured parquet.

Usage
-----
    python ocr_pipeline.py
    python ocr_pipeline.py --pdf data/OCR-data/3357.pdf
    python ocr_pipeline.py --pages 1-3 --debug
    python ocr_pipeline.py --no-preprocess

Output
------
    data/ocr_output/<id>.parquet                  structured depth intervals
    data/ocr_output/<id>_raw_pages.json           raw OCR text per page (for re-parsing)
    data/ocr_output/<id>_header.json              extracted well header metadata
    data/ocr_output/<id>_ocr_diagnostics.json     per-page dimensions and confidence
    data/ocr_output/debug/page_NNN_original.png       rendered page     (--debug)
    data/ocr_output/debug/page_NNN_preprocessed.png   after preprocessing (--debug)

Output schema (long-form, compatible with data_layer.load_samples()):
    dataset, borehole, depth_top, depth_base, depth,
    description, formation, x_rd, y_rd, page_number
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.ocr_config import (
    OCR_INPUT_DIR,
    OCR_OUTPUT_DIR,
    PDF_DPI,
    POPPLER_PATH,
    TESSERACT_CMD,
    TESSERACT_CONFIG,
    TESSERACT_LANG,
)
from scripts.ocr_parser import build_parse_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_PIXEL_WARN_THRESHOLD = 120_000_000
_PIXEL_MAX = 250_000_000


# ── Lazy imports ───────────────────────────────────────────────────────────


def _import_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        log.error(
            "opencv-python-headless is not installed. Run: pip install opencv-python-headless"
        )
        sys.exit(1)


def _import_pil():
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = _PIXEL_MAX
        return Image
    except ImportError:
        log.error("Pillow is not installed. Run: pip install Pillow")
        sys.exit(1)


def _import_pytesseract(tesseract_cmd: str):
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return pytesseract
    except ImportError:
        log.error("pytesseract is not installed. Run: pip install pytesseract")
        sys.exit(1)


def _import_pdf2image():
    try:
        from pdf2image import convert_from_path

        return convert_from_path
    except ImportError:
        log.error("pdf2image is not installed. Run: pip install pdf2image")
        sys.exit(1)


# ── Image preprocessing ────────────────────────────────────────────────────


def _deskew_angle(gray: np.ndarray) -> float:
    """Estimate page skew angle from near-horizontal Hough lines."""
    cv2 = _import_cv2()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) < 10:
                angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def preprocess_image(pil_img) -> object:
    """Preprocessing pipeline for borehole log scans.

    Steps (order matters):
    1. Grayscale
    2. Deskew  — rotate before denoising to avoid resampling artifacts
    3. Denoise — fastNlMeansDenoising
    4. Adaptive threshold — handles non-uniform background from log tracks
    5. Border removal — white-out thick black borders that mislead layout analysis
    """
    cv2 = _import_cv2()
    Image = _import_pil()

    img_np = np.array(pil_img.convert("L"))

    angle = _deskew_angle(img_np)
    if abs(angle) > 0.1:
        h, w = img_np.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img_np = cv2.warpAffine(
            img_np,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    img_np = cv2.fastNlMeansDenoising(
        img_np, h=10, templateWindowSize=7, searchWindowSize=21
    )

    img_np = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    border = 20
    img_np[:border, :] = 255
    img_np[-border:, :] = 255
    img_np[:, :border] = 255
    img_np[:, -border:] = 255

    return Image.fromarray(img_np)


# ── PDF rendering ──────────────────────────────────────────────────────────


def render_single_page(
    pdf_path: Path, page_num: int, dpi: int, poppler_path: Path
) -> list:
    """Render one PDF page (1-indexed) to a PIL Image. Returns empty list if beyond end."""
    convert = _import_pdf2image()
    return convert(
        str(pdf_path),
        dpi=dpi,
        first_page=page_num,
        last_page=page_num,
        poppler_path=str(poppler_path),
        use_cropbox=True,
    )


def _check_image_size(img, page_num: int) -> None:
    """Log image dimensions and warn if the page exceeds the large-image threshold."""
    w, h = img.size
    mp = w * h / 1_000_000
    log.info("  Page %d: %d × %d px (%.1f MP)", page_num, w, h, mp)
    if w * h > _PIXEL_WARN_THRESHOLD:
        log.warning(
            "  Page %d is large (%.1f MP) — consider lowering --dpi to reduce memory usage.",
            page_num,
            mp,
        )


# ── OCR ────────────────────────────────────────────────────────────────────


def ocr_page(pil_img, config: str, lang: str, tesseract_cmd: str) -> str:
    """Run Tesseract on a single PIL Image and return raw text."""
    pytesseract = _import_pytesseract(tesseract_cmd)
    return pytesseract.image_to_string(pil_img, lang=lang, config=config)


_LOW_CONF_THRESHOLD = 60  # diagnostic only — no text is filtered based on this


def ocr_page_diagnostics(
    pil_img, config: str, lang: str, tesseract_cmd: str, page_num: int, page_id: str
) -> dict:
    """Return per-page confidence distribution statistics (diagnostic only).

    conf == -1 rows are layout/non-word entries from Tesseract and are excluded
    from all statistics. No OCR text is altered or removed based on confidence.
    """
    pytesseract = _import_pytesseract(tesseract_cmd)
    data = pytesseract.image_to_data(
        pil_img, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME
    )
    words = data[(data["conf"] != -1) & (data["text"].str.strip() != "")]
    conf = words["conf"]
    n = len(conf)

    if n > 0:
        low_count = int((conf < _LOW_CONF_THRESHOLD).sum())
        stats = {
            "mean_confidence":          round(float(conf.mean()), 1),
            "std_confidence":           round(float(conf.std()), 1),
            "median_confidence":        round(float(conf.median()), 1),
            "min_confidence":           int(conf.min()),
            "max_confidence":           int(conf.max()),
            "low_confidence_word_count": low_count,
            "low_confidence_word_share": round(low_count / n, 3),
        }
    else:
        stats = {
            "mean_confidence":          None,
            "std_confidence":           None,
            "median_confidence":        None,
            "min_confidence":           None,
            "max_confidence":           None,
            "low_confidence_word_count": 0,
            "low_confidence_word_share": None,
        }

    return {
        "page_id":   page_id,
        "page_number": page_num,
        "word_count": n,
        **stats,
        "processed": True,
    }



def _is_appendix_page(text: str) -> bool:
    """Return True if any line in the page starts with 'appendix'."""
    for line in text.splitlines():
        if line.strip().lower().startswith("appendix"):
            return True
    return False


# ── Output ─────────────────────────────────────────────────────────────────


def build_output_df(result, borehole_id: str) -> pd.DataFrame:
    """Build a schema-compatible DataFrame from a ParseResult."""
    rows = []
    for iv in result.intervals:
        rows.append(
            {
                "dataset": "OCR",
                "borehole": borehole_id,
                "depth_top": iv.depth_top,
                "depth_base": iv.depth_base,
                "depth": (iv.depth_top + iv.depth_base) / 2.0,
                "description": iv.description,
                "formation": iv.formation,
                "x_rd": (
                    result.header.x_rd if result.header.x_rd is not None else np.nan
                ),
                "y_rd": (
                    result.header.y_rd if result.header.y_rd is not None else np.nan
                ),
                "page_number": iv.page_number,
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    df: pd.DataFrame,
    result,
    borehole_id: str,
    out_dir: Path,
    raw_page_objects: list[dict],
    diagnostics: list[dict],
    source_pdf: str,
    dpi: int,
    preprocessing_enabled: bool,
) -> None:
    """Write parquet, raw pages JSON, header JSON, and diagnostics JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{borehole_id}.parquet"
    df.to_parquet(parquet_path, index=False)
    log.info("Wrote %d intervals → %s", len(df), parquet_path)

    raw_path = out_dir / f"{borehole_id}_raw_pages.json"
    raw_doc = {
        "pdf_id": borehole_id,
        "source_pdf": Path(source_pdf).name,
        "pages": raw_page_objects,
    }
    raw_path.write_text(json.dumps(raw_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote raw OCR text → %s", raw_path)

    header_path = out_dir / f"{borehole_id}_header.json"
    header_path.write_text(
        json.dumps(asdict(result.header), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote header metadata → %s", header_path)

    diag_doc = {
        "pdf_id": borehole_id,
        "source_pdf": Path(source_pdf).name,
        "dpi": dpi,
        "preprocessing_enabled": preprocessing_enabled,
        "pages": diagnostics,
    }
    diag_path = out_dir / f"{borehole_id}_ocr_diagnostics.json"
    diag_path.write_text(json.dumps(diag_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote OCR diagnostics → %s", diag_path)

    if result.parse_warnings:
        log.warning("%d parse warnings:", len(result.parse_warnings))
        for w in result.parse_warnings:
            log.warning("  %s", w)


# ── Page range helper ──────────────────────────────────────────────────────


def _parse_page_bounds(spec: str | None) -> tuple[int, int | None]:
    """Parse '1-3' or '2' into (first, last) 1-indexed bounds. None means start from 1 with no upper limit."""
    if spec is None:
        return (1, None)
    parts = spec.split("-")
    first = int(parts[0])
    last = int(parts[1]) if len(parts) > 1 else first
    return (first, last)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tesseract OCR pipeline for NLOG borehole PDFs"
    )
    ap.add_argument(
        "--pdf",
        type=Path,
        default=OCR_INPUT_DIR / "3357.pdf",
        help="path to input PDF (default: data/OCR-data/3357.pdf)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OCR_OUTPUT_DIR,
        help="output directory (default: data/ocr_output/)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=PDF_DPI,
        help="rendering DPI — 200-300 is usually sufficient (default: %(default)s)",
    )
    ap.add_argument(
        "--pages",
        type=str,
        default=None,
        help="1-indexed page range to process, e.g. '1-5' or '2'",
    )
    ap.add_argument(
        "--no-preprocess",
        action="store_true",
        help="skip image preprocessing and send the rendered page directly to Tesseract",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="save original and preprocessed page images to out/debug/",
    )
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    if args.dpi > 300:
        log.warning(
            "--dpi %d is high and will produce very large images. 200-300 is usually sufficient.",
            args.dpi,
        )

    borehole_id = pdf_path.stem

    try:
        pytesseract = _import_pytesseract(TESSERACT_CMD)
        log.info("Tesseract version: %s", pytesseract.get_tesseract_version())
    except Exception as exc:
        log.error(
            "Tesseract not found or not configured correctly.\n"
            "  Install from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  Then update TESSERACT_CMD in scripts/ocr_config.py\n"
            "  Error: %s",
            exc,
        )
        sys.exit(1)

    if not POPPLER_PATH.exists():
        log.error(
            "Poppler bin directory not found: %s\n"
            "  Install from: https://github.com/oschwartz10612/poppler-windows/releases\n"
            "  Then update POPPLER_PATH in scripts/ocr_config.py",
            POPPLER_PATH,
        )
        sys.exit(1)

    debug_dir = args.out / "debug" if args.debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    first_page, last_page = _parse_page_bounds(args.pages)
    raw_page_objects: list[dict] = []
    diagnostics: list[dict] = []
    page_num = first_page
    appendix_check_start = first_page + 10  # skip appendix check for the first 10 pages

    with tqdm(desc="OCR pages") as pbar:
        while True:
            images = render_single_page(pdf_path, page_num, args.dpi, POPPLER_PATH)
            if not images:
                break

            img = images[0]
            _check_image_size(img, page_num)

            if debug_dir:
                img.save(debug_dir / f"page_{page_num:03d}_original.png")

            ocr_img = img if args.no_preprocess else preprocess_image(img)

            if debug_dir and not args.no_preprocess:
                ocr_img.save(debug_dir / f"page_{page_num:03d}_preprocessed.png")

            text = ocr_page(ocr_img, TESSERACT_CONFIG, TESSERACT_LANG, TESSERACT_CMD)

            if page_num >= appendix_check_start and _is_appendix_page(text):
                log.info("Appendix detected on page %d — stopping.", page_num)
                break

            page_id = f"{borehole_id}_p{page_num:03d}"
            diag = ocr_page_diagnostics(
                ocr_img, TESSERACT_CONFIG, TESSERACT_LANG, TESSERACT_CMD, page_num, page_id
            )
            raw_page_objects.append({"page_id": page_id, "page_number": page_num, "text": text})
            diagnostics.append(diag)
            pbar.update(1)
            page_num += 1

            if last_page is not None and page_num > last_page:
                break

    log.info("OCR complete (%d pages). Parsing …", len(raw_page_objects))

    raw_texts = [p["text"] for p in raw_page_objects]
    result = build_parse_result(raw_texts, source_pdf=str(pdf_path))
    log.info(
        "Parsed %d depth intervals, %d warnings.",
        len(result.intervals),
        len(result.parse_warnings),
    )

    df = build_output_df(result, borehole_id)
    save_outputs(
        df,
        result,
        borehole_id,
        out_dir=args.out,
        raw_page_objects=raw_page_objects,
        diagnostics=diagnostics,
        source_pdf=str(pdf_path),
        dpi=args.dpi,
        preprocessing_enabled=not args.no_preprocess,
    )

    log.info("Done. Output in: %s", args.out)


if __name__ == "__main__":
    main()
