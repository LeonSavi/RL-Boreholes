"""
ocr_parser.py
=============
Pure functions for parsing raw OCR text from Dutch NLOG borehole log PDFs
into structured depth-interval records.

No file I/O here — all functions are stateless and directly testable.

Dutch borehole log format (NLOG standard):
  - Depth intervals: "0.0 - 50.0 m" or "0,0 - 50,0" (comma as decimal separator)
  - Lithology descriptions in Dutch: "zandsteen", "klei", "kalk", etc.
  - Formation codes: same NLOG codes used in pull_data.py
  - Well header on page 1: name, operator, RD coordinates, total depth
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class WellHeader:
    borehole_id: str | None = None
    well_name: str | None = None
    operator: str | None = None
    x_rd: float | None = None
    y_rd: float | None = None
    total_depth_m: float | None = None
    source_pdf: str | None = None


@dataclass
class DepthInterval:
    depth_top: float
    depth_base: float
    description: str
    formation: str | None = None
    page_number: int | None = None


@dataclass
class ParseResult:
    header: WellHeader
    intervals: list[DepthInterval] = field(default_factory=list)
    raw_pages: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


# ── Patterns ───────────────────────────────────────────────────────────────

# Handles "0.0 - 50.0", "0,0-50,0", "0 - 50 m", en-dash variant
_DEPTH_INTERVAL_RE = re.compile(
    r"(\d{1,5}[.,]\d{0,2}|\d{1,5})"
    r"\s*[-–]\s*"
    r"(\d{1,5}[.,]\d{0,2}|\d{1,5})"
    r"\s*(?:m\b)?",
    re.IGNORECASE,
)

# NLOG formation codes present in the existing pull_data.py schema
_FORMATION_CODE_RE = re.compile(
    r"\b(NU|CK|KN|RN|RB|ZE|RO|NM|NL|DC|AT|SL|SG|SK)\b"
)

# Well name / borehole identifier
_WELL_NAME_RE = re.compile(
    r"(?:putnaam|boorput|well\s*name|borehole)[:\s]+([A-Z0-9\-/]+)",
    re.IGNORECASE,
)

# Operator
_OPERATOR_RE = re.compile(
    r"(?:operator|maatschappij)[:\s]+([A-Za-z0-9 &.,\-]+)",
    re.IGNORECASE,
)

# Dutch RD coordinates
_X_RD_RE = re.compile(r"(?:x[-_\s]?(?:rd|coordinaat)?)[:\s]+([\d.,]+)", re.IGNORECASE)
_Y_RD_RE = re.compile(r"(?:y[-_\s]?(?:rd|coordinaat)?)[:\s]+([\d.,]+)", re.IGNORECASE)

# Total depth
_TD_RE = re.compile(
    r"(?:total\s*depth|TD|eindiepte|totale\s*diepte)[:\s]+([\d.,]+)\s*m?",
    re.IGNORECASE,
)

# Conservative OCR corrections for known formation-code misreads only.
# Applied to a copy of the description; the original is never modified.
_FORMATION_OCR_FIXES: list[tuple[str, str]] = [
    ("N1", "NL"),
    ("R N", "RN"),
    ("N U", "NU"),
]

# ── Pure functions ─────────────────────────────────────────────────────────

def _fix_formation_ocr(text: str) -> str:
    """Apply conservative corrections for known OCR misreads in formation codes."""
    for wrong, right in _FORMATION_OCR_FIXES:
        text = text.replace(wrong, right)
    return text


def normalise_depth(raw: str) -> float:
    """Convert a raw depth string to float, handling Dutch comma-decimal notation."""
    cleaned = raw.strip().replace(",", ".")
    return float(cleaned)


def merge_continuation_lines(lines: list[str]) -> list[str]:
    """Join continuation lines onto their preceding depth-interval line.

    - Empty lines are skipped.
    - A line starting with a depth interval begins a new record.
    - Any other non-empty line is appended (with a space) to the previous record.
    - Leading non-depth text before the first interval is kept as its own record.
    """
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _DEPTH_INTERVAL_RE.match(stripped):
            merged.append(stripped)
        elif merged:
            merged[-1] = merged[-1].rstrip() + " " + stripped
        else:
            merged.append(stripped)
    return merged


def parse_header(pages: list[str], source_pdf: str) -> WellHeader:
    """Extract well header metadata from the first two pages of raw OCR text."""
    header = WellHeader(source_pdf=source_pdf)
    search_text = "\n".join(pages[:2])

    m = _WELL_NAME_RE.search(search_text)
    if m:
        header.well_name = m.group(1).strip()

    m = _OPERATOR_RE.search(search_text)
    if m:
        header.operator = m.group(1).strip()

    m = _X_RD_RE.search(search_text)
    if m:
        try:
            header.x_rd = normalise_depth(m.group(1))
        except ValueError:
            pass

    m = _Y_RD_RE.search(search_text)
    if m:
        try:
            header.y_rd = normalise_depth(m.group(1))
        except ValueError:
            pass

    m = _TD_RE.search(search_text)
    if m:
        try:
            header.total_depth_m = normalise_depth(m.group(1))
        except ValueError:
            pass

    return header


def parse_depth_intervals(pages: list[str]) -> tuple[list[DepthInterval], list[str]]:
    """Extract depth-interval records from all pages.

    Returns a tuple of (intervals, warnings). Skips:
    - Intervals where depth_top >= depth_base (OCR error)
    - Intervals where depth_base - depth_top > 500 m (implausibly thick)
    """
    intervals: list[DepthInterval] = []
    warnings: list[str] = []

    for page_num, page_text in enumerate(pages, start=1):
        lines = merge_continuation_lines(page_text.splitlines())
        for line in lines:
            m = _DEPTH_INTERVAL_RE.match(line.lstrip())
            if not m:
                continue

            try:
                top = normalise_depth(m.group(1))
                base = normalise_depth(m.group(2))
            except ValueError:
                warnings.append(f"p{page_num}: could not parse depths in: {line!r}")
                continue

            if top >= base:
                warnings.append(f"p{page_num}: skipped inverted interval {top}-{base}")
                continue
            if base - top > 500:
                warnings.append(f"p{page_num}: skipped implausible interval {top}-{base}")
                continue

            description = line[m.end():].strip()

            description_corrected = _fix_formation_ocr(description)
            formation_match = _FORMATION_CODE_RE.search(description_corrected)
            formation = formation_match.group(1) if formation_match else None

            intervals.append(DepthInterval(
                depth_top=top,
                depth_base=base,
                description=description,
                formation=formation,
                page_number=page_num,
            ))

    return intervals, warnings


def deduplicate_intervals(intervals: list[DepthInterval]) -> list[DepthInterval]:
    """Remove duplicate intervals with identical top/base depths.

    Duplicates arise when Tesseract reads a multi-column log layout and
    encounters the same row twice (e.g., depth printed in both columns).
    """
    seen: set[tuple[float, float]] = set()
    unique: list[DepthInterval] = []
    for iv in intervals:
        key = (iv.depth_top, iv.depth_base)
        if key not in seen:
            seen.add(key)
            unique.append(iv)
    return unique


def build_parse_result(
    raw_pages: list[str],
    source_pdf: str,
) -> ParseResult:
    """Run the full parse pipeline and return a ParseResult."""
    header = parse_header(raw_pages, source_pdf)
    intervals, warnings = parse_depth_intervals(raw_pages)
    intervals = deduplicate_intervals(intervals)
    return ParseResult(
        header=header,
        intervals=intervals,
        raw_pages=raw_pages,
        parse_warnings=warnings,
    )
