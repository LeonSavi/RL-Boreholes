"""
Pull LILY + NLOG into ONE unified Parquet (`samples.parquet`) with a
`dataset` column. Expanded feature set + all Tier 1+2 sanity checks.

v5 change log
-------------
  * Reads NLOG boreholes.xlsx for location + result metadata
  * New column `hc_result` — hydrocarbon discovery label
    ('gas' | 'oil' | 'gas+oil' | 'dry' | 'show' | 'water' | ...)
  * details.json becomes fallback when boreholes.xlsx lacks a well

Output schema
-------------
dataset          str       'LILY' | 'NLOG'
borehole         str       well/site identifier
depth            float     metres below deposition surface
measurement      str       rhob | gr_api | dt_us_ft | ...
value            float     cleaned measurement
rock_type        str       coarse rock-type
rock_type_fine   str       sub-formation-aware rock-type
formation        string    NLOG formation code (e.g. 'RO') or <NA>
strat_unit       string    NLOG sub-unit code (e.g. 'ROSLV') or <NA>
period           string    e.g. 'Permian' or <NA>
era              string    'Cenozoic' | 'Mesozoic' | 'Paleozoic' or <NA>
lith_principal   string    LILY principal lithology or <NA>
x_rd             float     NLOG x (Dutch RD) or NaN
y_rd             float     NLOG y (Dutch RD) or NaN
location_type    string    'onshore' | 'offshore' | <NA>
hc_result        string    hydrocarbon label from boreholes.xlsx or <NA>
hc_discovery     boolean   True=gas|oil|gas+oil, False=dry|water|abandoned,
                           <NA>=show|other|unknown (deliberately three-way)
field_name       string    NLOG field name (e.g. 'Groningen') or <NA>
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

logging.getLogger("lasio").setLevel(logging.ERROR)


LILY_DIR            = Path("data/lily")
NLOG_DIR            = Path("data/nlog/nlog_scrape")
NLOG_BOREHOLES_XLSX = Path("data/nlog/boreholes.xlsx")
DEFAULT_OUT         = Path("data/clean")

DEPTH_STEP_M = 1.0
FT_TO_M      = 0.3048

NLOG_FORMATIONS = ["NU", "CK", "KN", "RN", "RB", "ZE", "RO",
                   "NM", "NL", "DC", "AT", "SL", "SG", "SK"]

NLOG_FORMATION_TO_ROCK = {
    "NU": "clay",       "CK": "chalk",     "KN": "claystone",
    "RN": "claystone",  "RB": "sandstone",
    # ZE (Zechstein) is DELIBERATELY 'other' — Zechstein as a whole is
    # mixed (~50% halite, 20% anhydrite, 25% carbonate, 5% claystone).
    # Rows with only 'ZE' formation-level code but no sub-unit cannot
    # safely be assigned to any one lithology. Only stratUnitId-level
    # codes (ZEZ1H=halite, ZEZ1A=anhydrite, ZEZ1C=carbonate) produce
    # typed rows; ambiguous Zechstein intervals drop to 'other' and get
    # filtered out of the simulator fit.
    "ZE": "other",
    "RO": "sandstone",  "NM": "clay",      "NL": "clay",
    "DC": "claystone",  "AT": "claystone", "SL": "claystone",
    "SG": "claystone",  "SK": "claystone",
}

NLOG_FORMATION_TO_PERIOD = {
    "NU": "Neogene",         "NM": "Neogene",        "NL": "Neogene",
    "CK": "Cretaceous",      "KN": "Cretaceous",
    "SL": "Jurassic",        "SG": "Jurassic",        "SK": "Jurassic",
    "AT": "Triassic",        "RN": "Triassic",        "RB": "Triassic",
    "ZE": "Permian",         "RO": "Permian",
    "DC": "Carboniferous",
}
PERIOD_TO_ERA = {
    "Neogene": "Cenozoic", "Paleogene": "Cenozoic",
    "Cretaceous": "Mesozoic", "Jurassic": "Mesozoic", "Triassic": "Mesozoic",
    "Permian": "Paleozoic", "Carboniferous": "Paleozoic", "Devonian": "Paleozoic",
}

STRAT_UNIT_TO_FINE_ROCK = {
    # ────────────────────────────────────────────────────────────────────
    # REFINED rock_type_fine scheme (v5).
    #
    # The earlier flat scheme lumped too many petrophysically-distinct
    # lithologies.  Within "sandstone" the gamma-ray ranged 21-113 API;
    # within "claystone" it ranged 31-137 API; within "halite" 6-70 API.
    # These spreads made the autoencoder unable to reconstruct gr_api.
    #
    # Split logic (geological, not gamma-threshold-arbitrary):
    #
    #   sandstone_clean   — reservoir-quality quartz-dominant sandstones
    #                       (Slochteren, post-rift marine sands).  Low Vsh,
    #                       gamma ~20-50 API, high porosity when undeformed.
    #   sandstone_shaly   — clay-rich sandstones (Silverpit, Buntsandstein,
    #                       fluvial coal-measure sands).  Gamma ~60-110 API.
    #
    #   claystone_cool    — marine marls, nannofossil clays, clean claystones.
    #                       Gamma ~40-80 API.  No organic enrichment, no
    #                       potassium feldspar.
    #   claystone_hot     — organic-rich or K-rich shales (Carboniferous coal
    #                       measures, Rotliegend Ten Boer, Posidonia-type).
    #                       Gamma ~80-150+ API.  Often source rocks.
    #
    #   halite_pure       — confirmed monomineralic halite members (H-suffix
    #                       codes from Zechstein). Gamma ~5-15 API.
    #   halite            — retained as fallback for mixed/unclear halite.
    #
    #   dolomite          — Zechstein carbonate members and Muschelkalk
    #                       dolomites (C-suffix Zechstein codes + RN carbs).
    #                       Dolomitic: higher PEF (3.1), distinct NPHI
    #                       signature vs limestone.
    #   carbonate         — retained as fallback for unclear carbonate.
    #
    # The "fallback" versions (claystone, sandstone, halite, carbonate)
    # are what pull_data uses when the sub-unit code is only 2 or 3 chars
    # (too vague to assign a refined type).  Keeps behaviour safe.
    # ────────────────────────────────────────────────────────────────────

    # ─── Rotliegend (RO) — Permian, gas reservoir interval ──────────
    "ROSL":   "sandstone_clean",   # Slochteren generic
    "ROSLU":  "sandstone_clean",   # Upper Slochteren
    "ROSLV":  "sandstone_clean",   # Lower Slochteren / Volpriehausen
    "ROSS":   "sandstone_shaly",   # Silverpit sandier parts — muddier than Slochteren
    "ROSSF":  "claystone_hot",     # Silverpit fine-grained
    "ROCLT":  "claystone_hot",     # Ten Boer Claystone — seal rock, gamma ~100
    "ROCL":   "claystone_hot",     # Rotliegend claystones generally

    # ─── Zechstein (ZE) — Permian, evaporites ───────────────────────
    # (group-only codes ZEZ1, ZEZ2, ... deliberately absent — they drop
    #  to 'other' via the formation fallback to avoid contamination)
    "ZEZ1C":  "dolomite",          # Werra Carbonate
    "ZEZ1A":  "anhydrite",
    "ZEZ1H":  "halite_pure",       # Werra Halite
    "ZEZ2C":  "dolomite",          # Stassfurt (Hauptdolomit) — type dolomite
    "ZEZ2A":  "anhydrite",         # Basal Anhydrite
    "ZEZ2H":  "halite_pure",
    "ZEZ3C":  "dolomite",
    "ZEZ3A":  "anhydrite",
    "ZEZ3H":  "halite_pure",
    "ZEZ4A":  "anhydrite",
    "ZEZ4H":  "halite_pure",
    "ZESA":   "anhydrite",         # Z Anhydrite unit — ~63k rows
    "ZESAU":  "anhydrite",
    "ZESAL":  "anhydrite",
    "ZEUC":   "dolomite",          # Upper Carbonate

    # ─── Buntsandstein (RB) — Triassic, continental sands + mud ─────
    "RBM":    "sandstone_shaly",   # Main Buntsandstein — fluvial, muddy
    "RBMH":   "sandstone_shaly",   # Hardegsen
    "RBMV":   "sandstone_shaly",   # Volpriehausen (within Bunt)
    "RBMD":   "sandstone_shaly",   # Detfurth
    "RBSH":   "claystone_hot",     # Solling Claystone — continental red-beds
    "RBSHS":  "claystone_hot",
    "RBSHM":  "claystone_hot",     # seen in data: mean=92 API, hot
    "RBSHR":  "claystone_hot",     # seen in data: mean=94 API, hot

    # ─── Muschelkalk / Keuper (RN) — Triassic carbonates + clay ─────
    "RNRO":   "claystone",         # Röt Formation — mixed
    "RNROC":  "dolomite",          # Röt Carbonate
    "RNROE":  "anhydrite",         # Röt Evaporite
    "RNMU":   "dolomite",          # Muschelkalk — carbonate (dolomitic)
    "RNMUE":  "anhydrite",
    "RNMUC":  "dolomite",
    "RNSOC":  "dolomite",          # Solling Carbonate (probably)
    "RNKPU":  "claystone",         # Keuper — continental red/green mud
    "RNKPL":  "claystone",
    "RNKP":   "claystone",
    "RNKPS":  "claystone",

    # ─── Chalk (CK) — Late Cretaceous ───────────────────────────────
    "CKEK":   "chalk",             # Ekofisk — classic reservoir chalk
    "CKTX":   "chalk",             # Texel
    "CKGR":   "chalk",             # Ommelanden + others

    # ─── Rijnland (KN) — Early Cretaceous marine ────────────────────
    "KNNC":   "claystone",         # seen in data: mean=78 API, moderate
    "KNNS":   "sandstone_shaly",   # Rijnland marine sands, muddier than reservoir-grade
    "KNGL":   "claystone_cool",    # Vlieland claystone — low gamma in data
    "KNGLU":  "claystone_cool",    # seen in data: mean=56 API — clearly cool
    "KNGLL":  "claystone_cool",    # seen in data: mean=74 API — borderline, treat as cool

    # ─── Altena (AT) — Early Jurassic marine shale ──────────────────
    "ATAL":   "claystone_hot",     # Aalburg — rich shale, seen at mean=85
    "ATWDL":  "claystone_cool",    # Werkendam Lower
    "ATWDU":  "claystone_cool",    # Werkendam Upper
    "ATRT":   "claystone",         # Altena Röt mixed

    # ─── Schieland / Germanic Triassic (SL, SG) — Upper Jurassic ────
    "SLDNA":  "claystone_cool",    # Delfland — seen at mean=76, moderate
    "SLDNR":  "claystone_cool",
    "SLDND":  "claystone_cool",
    "SLDN":   "claystone_cool",
    "SLCL":   "claystone_cool",
    "SGKI":   "claystone_hot",     # Kimmeridge — classic source rock, hot

    # ─── Cenozoic (NU, NM, NL) — Tertiary/Quaternary ────────────────
    "NUBA":   "clay",              # Breda
    "NUIE":   "clay",              # IJsselmeer
    "NLFF":   "clay",              # Lower North Sea
    "NMDO":   "clay",              # Dongen
    "NUOT":   "clay",              # Oosterhout
    "NUMS":   "clay",              # Middle North Sea
    "NLLFC":  "clay",              # Landen
    "NMRF":   "clay",              # Rupel
    "NMRFC":  "clay",

    # ─── Carboniferous (DC) — coal measures ─────────────────────────
    "DCCU":   "claystone_hot",     # Upper Carboniferous — coal-bearing = very hot gamma
    "DCCR":   "claystone_hot",     # Caumer subgroup
    "DCDT":   "claystone_hot",
    "DCDG":   "claystone_hot",
    "DCHL":   "claystone_hot",     # Hellevoetsluis
    "DCGE":   "claystone_hot",     # Geverik — organic-rich shale
}


@dataclass
class CurveSpec:
    aliases: tuple
    bounds:  tuple
    transform: callable | None = None
    gradient_lim: float | None = None


NLOG_PRIMARY: dict[str, CurveSpec] = {
    "rhob":     CurveSpec(("RHOB",),                       (1.5, 3.3), gradient_lim=0.5),
    "gr_api":   CurveSpec(("GR",),                         (0, 400)),
    "nphi":     CurveSpec(("NPHI",),                       (-0.15, 0.80)),
    "dt_us_ft": CurveSpec(("DTC", "DTCO", "DTP", "DT"),    (40, 250), gradient_lim=30.0),
    "pef":      CurveSpec(("PEF", "PE"),                   (1.0, 10.0)),
    "cali_in":  CurveSpec(("CALI", "CAL"),                 (4.0, 30.0)),
    "sp_mv":    CurveSpec(("SP",),                         (-200.0, 200.0)),
    "res_deep_log":  CurveSpec(("ILD", "LLD", "RT", "RD"),
                                bounds=(-1.0, 4.0),
                                transform=lambda v: np.log10(v) if v > 0 else np.nan),
    "res_shal_log":  CurveSpec(("ILM", "LLS", "RS", "SFLU", "MSFL"),
                                bounds=(-1.0, 4.0),
                                transform=lambda v: np.log10(v) if v > 0 else np.nan),
}
NLOG_QUALITY: dict[str, CurveSpec] = {
    "drho":     CurveSpec(("DRHO",), (-0.3, 0.3)),
}
NLOG_DERIVED: dict[str, CurveSpec] = {
    "vsh":      CurveSpec(("VSH", "VSHALE"),   (0.0, 1.0)),
    "vcl":      CurveSpec(("VCL", "VCLAY"),    (0.0, 1.0)),
    "vsnd":     CurveSpec(("VSND", "VSAND"),   (0.0, 1.0)),
    "vdol":     CurveSpec(("VDOL",),           (0.0, 1.0)),
    "vsilt":    CurveSpec(("VSILT",),          (0.0, 1.0)),
    "phie":     CurveSpec(("PHIE", "PHIT"),    (0.0, 0.45)),
    "sw":       CurveSpec(("SW", "SWE"),       (0.0, 1.0)),
}
ALL_NLOG_SPECS = {**NLOG_PRIMARY, **NLOG_QUALITY, **NLOG_DERIVED}

SONIC_SHEAR_MNEMONICS = {"DTS", "DTSM", "DTS_FAST", "DTS_SLOW"}

OUTLIER_MAD_MULTIPLIER     = 5.0
FLAT_INTERVAL_MIN_SAMPLES  = 20
MAX_EXPECTED_GAP_M         = 50.0

LITH_PATTERNS = [
    ("chalk",       ["chalk"]),
    ("sandstone",   ["sandstone", "sand"]),
    ("siltstone",   ["siltstone"]),
    ("claystone",   ["claystone"]),
    ("mudstone",    ["mudstone", "mud"]),
    ("anhydrite",   ["anhydrite"]),
    ("halite",      ["halite", "salt"]),
    ("dolomite",    ["dolomite", "dolostone"]),
    ("limestone",   ["limestone", "packstone", "wackestone", "grainstone"]),
    ("diatom_ooze", ["diatom ooze", "diatomite"]),
    ("nanno_ooze",  ["nannofossil ooze", "calcareous ooze", "foram ooze"]),
    ("basalt",      ["basalt", "gabbro"]),
    ("clay",        ["clay"]),
]

def classify_lithology(principal: str) -> str:
    if not isinstance(principal, str):
        return "other"
    p = principal.lower()
    for label, keywords in LITH_PATTERNS:
        for kw in keywords:
            if kw in p:
                return label
    return "other"


def classify_strat_unit(strat_unit: str | None, formation: str | None) -> str:
    if isinstance(strat_unit, str) and strat_unit:
        best_match = None
        best_len = 0
        for prefix, rock in STRAT_UNIT_TO_FINE_ROCK.items():
            if strat_unit.startswith(prefix) and len(prefix) > best_len:
                best_match = rock
                best_len = len(prefix)
        if best_match is not None:
            return best_match
    if formation and formation in NLOG_FORMATION_TO_ROCK:
        return NLOG_FORMATION_TO_ROCK[formation]
    return "other"


@dataclass
class PullReport:
    lily_rows_in:      int = 0
    lily_rows_out:     int = 0
    lily_rows_dropped: dict = field(default_factory=lambda: defaultdict(int))
    lily_nonfinite:    int = 0

    nlog_wells_in:         int = 0
    nlog_wells_with_las:   int = 0
    nlog_wells_with_strat: int = 0
    nlog_wells_used:       int = 0
    nlog_rows_out:         int = 0
    nlog_curve_counts:     Counter = field(default_factory=Counter)
    nlog_shear_wells:      list = field(default_factory=list)
    nlog_bad_index:        list = field(default_factory=list)
    nlog_unit_issues:      list = field(default_factory=list)
    nlog_feet_wells:       list = field(default_factory=list)
    nlog_empty_strat:      list = field(default_factory=list)
    nlog_corrupt_json:     list = field(default_factory=list)
    formation_hits:        Counter = field(default_factory=Counter)
    strat_unit_hits:       Counter = field(default_factory=Counter)
    strat_unit_unknown:    Counter = field(default_factory=Counter)
    strat_bad_intervals:   int = 0
    strat_overlaps:        int = 0
    strat_unknown_prefix:  Counter = field(default_factory=Counter)
    # counts of Zechstein rows with only group-level codes (dropped to 'other')
    ze_group_only:         Counter = field(default_factory=Counter)
    clipping_counts:       dict = field(default_factory=lambda: defaultdict(int))
    nonfinite_in_las:      int = 0
    per_feature_rows:      Counter = field(default_factory=Counter)
    per_feature_wells:     dict = field(default_factory=lambda: defaultdict(set))
    alias_hits:            Counter = field(default_factory=Counter)

    location_hits:         Counter = field(default_factory=Counter)
    coords_missing:        int = 0
    details_corrupt:       list = field(default_factory=list)

    # boreholes.xlsx metadata
    xlsx_rows:             int = 0
    xlsx_matched:          int = 0
    xlsx_unmatched:        list = field(default_factory=list)
    xlsx_columns_used:     dict = field(default_factory=dict)
    result_hits:           Counter = field(default_factory=Counter)
    discovery_hits:        Counter = field(default_factory=Counter)

    duplicate_rows:        int = 0
    outlier_wells:         dict = field(default_factory=lambda: defaultdict(list))
    gradient_spikes:       dict = field(default_factory=lambda: defaultdict(int))
    flat_intervals:        dict = field(default_factory=lambda: defaultdict(int))
    large_depth_gaps:      int = 0

    def write(self, path: Path) -> None:
        L: list = []
        add = L.append
        add("=" * 78)
        add(" PULL + SANITY REPORT  (v5: boreholes.xlsx + hc_result)")
        add("=" * 78)

        add("\n## LILY")
        add(f"  rows read                         {self.lily_rows_in:>14,}")
        add(f"  rows kept                         {self.lily_rows_out:>14,}"
            f"  ({100*self.lily_rows_out/max(self.lily_rows_in,1):.1f}%)")
        if self.lily_rows_dropped:
            add("  dropped because:")
            for reason, n in sorted(self.lily_rows_dropped.items(),
                                     key=lambda x: -x[1]):
                add(f"    {reason:<45s} {n:>12,}")

        add("\n## NLOG — file-level (Tier 1)")
        add(f"  wells scanned                     {self.nlog_wells_in:>14,}")
        add(f"    with LAS                        {self.nlog_wells_with_las:>14,}")
        add(f"    with strat.json                 {self.nlog_wells_with_strat:>14,}")
        add(f"    used                            {self.nlog_wells_used:>14,}")
        add(f"    feet-depth auto-converted       {len(self.nlog_feet_wells):>14,}")
        add(f"  rows written                      {self.nlog_rows_out:>14,}")

        add(f"\n## boreholes.xlsx (location + hc_result)")
        add(f"  rows read                         {self.xlsx_rows:>14,}")
        add(f"  matched to NLOG folders           {self.xlsx_matched:>14,}")
        add(f"  unmatched NLOG folders            {len(self.xlsx_unmatched):>14,}")
        if self.xlsx_columns_used:
            add(f"  columns used:")
            for role, col in self.xlsx_columns_used.items():
                add(f"    {role:<16s} {col!r}")

        add(f"\n## Location metadata (xlsx → details.json fallback)")
        add(f"  coords missing                    {self.coords_missing:>14,}")
        if self.details_corrupt:
            add(f"  corrupt details.json ({len(self.details_corrupt)} wells)")
        for loc_type, n in sorted(self.location_hits.items(), key=lambda x: -x[1]):
            add(f"    {loc_type:<16s} wells={n:>6,}")

        if self.result_hits:
            add(f"\n## Hydrocarbon result labels (from boreholes.xlsx)")
            for r, n in sorted(self.result_hits.items(), key=lambda x: -x[1]):
                add(f"    {r:<16s} wells={n:>6,}")

        if self.discovery_hits:
            add(f"\n## Pooled discovery label (hc_discovery)")
            add(f"  positive = gas | oil | gas+oil")
            add(f"  negative = dry | water | abandoned")
            add(f"  unknown  = show | other | unlabelled")
            for label in ("positive", "negative", "unknown"):
                n = self.discovery_hits.get(label, 0)
                add(f"    {label:<10s} wells={n:>6,}")

        add(f"\n## Feature availability in NLOG output")
        for meas in (list(NLOG_PRIMARY.keys()) +
                     list(NLOG_QUALITY.keys()) +
                     list(NLOG_DERIVED.keys())):
            n_rows  = self.per_feature_rows.get(meas, 0)
            n_wells = len(self.per_feature_wells.get(meas, set()))
            add(f"    {meas:<16s} wells={n_wells:>6,}   rows={n_rows:>10,}")

        if self.alias_hits:
            add(f"\n## Alias hits (how often each raw mnemonic was used)")
            for (out_name, alias), n in sorted(self.alias_hits.items(),
                                                key=lambda x: -x[1])[:30]:
                add(f"    {out_name:<14s} ← {alias:<10s} {n:>8}")

        add(f"\n## NLOG formation hits (interval count)")
        for f, n in sorted(self.formation_hits.items(), key=lambda x: -x[1]):
            add(f"    {f:<4s} {n:>10,}")

        add(f"\n## Sub-unit (stratUnitId) hits — mapped")
        for su, n in sorted(self.strat_unit_hits.items(), key=lambda x: -x[1])[:30]:
            rock = classify_strat_unit(su, su[:2] if len(su) >= 2 else None)
            add(f"    {su:<10s} → {rock:<12s} {n:>10,}")

        if self.strat_unit_unknown:
            add(f"\n## Sub-unit codes NOT in dictionary (fell back to formation)")
            total_unknown = sum(self.strat_unit_unknown.values())
            add(f"  total rows affected: {total_unknown:,}")
            for su, n in sorted(self.strat_unit_unknown.items(),
                                 key=lambda x: -x[1])[:20]:
                add(f"    {su:<10s} {n:>10,}")

        if self.strat_unknown_prefix:
            add(f"\n## Strat 2-char prefixes not in NLOG_FORMATIONS (dropped)")
            for prefix, n in sorted(self.strat_unknown_prefix.items(),
                                     key=lambda x: -x[1])[:10]:
                add(f"    {prefix:<12s} {n:>10,}")

        if self.ze_group_only:
            add(f"\n## Zechstein group-level codes (reclassified to 'other')")
            add(f"  these rows have only ZEZ1/ZEZ2/... (no H/A/C member suffix)")
            add(f"  so we cannot assign halite/anhydrite/carbonate safely")
            total = sum(self.ze_group_only.values())
            add(f"  total rows affected: {total:,}")
            for su, n in sorted(self.ze_group_only.items(),
                                 key=lambda x: -x[1])[:15]:
                add(f"    {su:<10s} {n:>10,}")

        add(f"\n## Curves seen in NLOG LAS files (top 25)")
        for curve, n in self.nlog_curve_counts.most_common(25):
            add(f"    {curve:<12s} {n:>8}")

        add(f"\n## Clipping events (values outside physical range — dropped)")
        for (curve, side), n in sorted(self.clipping_counts.items(),
                                        key=lambda x: -x[1])[:20]:
            add(f"    {curve:<16s} {side:<14s} {n:>14,}")

        add(f"\n## Tier 2 — statistical sanity")
        add(f"  duplicate rows removed            {self.duplicate_rows:>14,}")
        add(f"  large depth gaps flagged          {self.large_depth_gaps:>14,}")

        if self.gradient_spikes:
            add(f"\n  implausible gradients (dropped):")
            for meas, n in self.gradient_spikes.items():
                add(f"    {meas:<14s} {n:>12,}")
        if self.flat_intervals:
            add(f"\n  flat (stuck-tool) intervals (dropped):")
            for meas, n in self.flat_intervals.items():
                add(f"    {meas:<14s} {n:>12,}")

        if self.outlier_wells:
            add(f"\n  population-outlier wells "
                f"(> {OUTLIER_MAD_MULTIPLIER}× MAD from corpus median):")
            for key, wells in self.outlier_wells.items():
                add(f"    {key:<22s} ({len(wells)} wells)")
                for bh in wells[:5]:
                    add(f"        {bh}")

        path.write_text("\n".join(L))
        print(f"\n→ report written to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# boreholes.xlsx loader
# ─────────────────────────────────────────────────────────────────────────────
#
# Actual schema from NLOG export (2026-04, 6698 rows × 61 cols):
#   ID (matches folder names):   'Borehole name'  (e.g. 'A05-01')
#   RD coordinates:              'X Dutch National Grid', 'Y Dutch National Grid'
#   Onshore/offshore:            'On offshore'   values: 'ON' | 'OFF'
#   Discovery result:            'Result code'   ('DRY', 'GSS', 'GAS', 'OIL', ...)
#                                'Result'        (human-readable label)
#   Field (if in a named field): 'Field name'
#   Objective (why drilled):     'Borehole objective code'  ('EXP-HC', ...)
#
# NLOG Result codes (verified from the sample + NLOG reference):
#   DRY  = dry hole
#   GAS  = gas discovery
#   OIL  = oil discovery
#   GOD  = gas + oil discovery
#   GSS  = gas shows (traces, not commercial)
#   OSS  = oil shows
#   HCS  = hydrocarbon shows (unspecified)
#   WAT  = water-bearing
#   SUS  = suspended / abandoned
#   Other codes fall through to 'other'.

# Map NLOG Result code -> simplified hc_result category
NLOG_RESULT_CODE_MAP = {
    "DRY":  "dry",
    "GAS":  "gas",
    "OIL":  "oil",
    "GOD":  "gas+oil",
    "GAS+OIL": "gas+oil",
    "GSS":  "show",       # gas shows
    "OSS":  "show",       # oil shows
    "HCS":  "show",       # hydrocarbon shows generic
    "WAT":  "water",
    "SUS":  "abandoned",
    "ABD":  "abandoned",
    "PNA":  "abandoned",  # plugged & abandoned
}


# Collapse the seven-way hc_result into a three-way discovery label for
# downstream modelling.  Positive = commercial hydrocarbon found (any fluid).
# Negative = well drilled and explicitly didn't yield hydrocarbons.
# Unknown (NaN) = shows (uninformative), other (unmapped), or missing label.
#
# Rationale for pooling gas/oil/gas+oil:
#   - same trap mechanics (reservoir + seal + closure), different fluid
#   - Dutch subsurface has very few oil-only wells (~1-3%), so splitting
#     loses positive samples to no statistical benefit
#   - Task 1's autoencoder learns "prospective rock" signature, not fluid
#     classification
HC_RESULT_POSITIVE = {"gas", "oil", "gas+oil"}
HC_RESULT_NEGATIVE = {"dry", "water", "abandoned"}
# explicitly uninformative: {"show", "other"} -> NaN


def _hc_discovery(hc_result: str | None) -> bool | None:
    if hc_result is None:
        return None
    if hc_result in HC_RESULT_POSITIVE:
        return True
    if hc_result in HC_RESULT_NEGATIVE:
        return False
    return None  # 'show', 'other' — deliberately three-way


def _safe_float(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _classify_on_offshore(v) -> str | None:
    """'ON' -> 'onshore', 'OFF' -> 'offshore'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    if s == "ON":
        return "onshore"
    if s == "OFF":
        return "offshore"
    return None


def _classify_result_code(code, label) -> str | None:
    """Map 'Result code' to hc_result category.  Falls back to parsing
    the 'Result' label if the code is missing or unknown."""
    # first try the code
    if code is not None and not (isinstance(code, float) and np.isnan(code)):
        s = str(code).strip().upper()
        if s in NLOG_RESULT_CODE_MAP:
            return NLOG_RESULT_CODE_MAP[s]

    # fall back to the label string
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    t = str(label).strip().lower()
    if not t or t in ("nan", "none", "-", "unknown"):
        return None
    has_gas = "gas" in t
    has_oil = "oil" in t or "olie" in t
    if has_gas and has_oil:
        return "gas+oil"
    if has_gas:
        return "show" if "show" in t else "gas"
    if has_oil:
        return "show" if "show" in t else "oil"
    if "dry" in t or "droog" in t:
        return "dry"
    if "water" in t:
        return "water"
    if "abandon" in t or "opgegeven" in t or "suspended" in t:
        return "abandoned"
    if "show" in t or "spoor" in t:
        return "show"
    return "other"


def _load_boreholes_metadata(
    xlsx_path: Path, report: PullReport,
) -> dict[str, dict]:
    """Load boreholes.xlsx into dict[well_id -> metadata].

    Returned dict is keyed by `Borehole name` with case/separator variants
    aliased so folder names on disk match regardless of capitalisation.

    metadata keys: x_rd, y_rd, location_type, hc_result,
                   result_code_raw, field_name, objective
    """
    if not xlsx_path.exists():
        print(f"  [skip] boreholes.xlsx not at {xlsx_path}")
        return {}

    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"  [warn] failed to read {xlsx_path}: {e}")
        return {}

    report.xlsx_rows = len(df)
    print(f"  loaded {xlsx_path.name}: {len(df)} rows, {len(df.columns)} cols")

    # Exact column names from the NLOG export (case-sensitive).
    EXPECTED = {
        "id":          "Borehole name",
        "x":           "X Dutch National Grid",
        "y":           "Y Dutch National Grid",
        "onshore":     "On offshore",
        "result_code": "Result code",
        "result_label": "Result",
        "field":       "Field name",
        "objective":   "Borehole objective code",
    }

    # Verify each expected column exists; warn about missing ones
    missing = [col for col in EXPECTED.values() if col not in df.columns]
    if missing:
        print(f"  [warn] boreholes.xlsx missing expected columns: {missing}")
        print(f"  available columns: {list(df.columns)[:20]} ...")

    report.xlsx_columns_used = {
        role: (col if col in df.columns else None)
        for role, col in EXPECTED.items()
    }

    id_col = EXPECTED["id"]
    if id_col not in df.columns:
        print(f"    [error] cannot find {id_col!r} column; boreholes.xlsx "
              f"metadata unavailable.  Falling back to details.json per well.")
        return {}

    wells_meta: dict[str, dict] = {}
    for _, row in df.iterrows():
        raw_id = row[id_col]
        if pd.isna(raw_id):
            continue
        bh = str(raw_id).strip()
        if not bh:
            continue

        def g(role: str):
            col = EXPECTED[role]
            return row[col] if col in df.columns else None

        hc_result = _classify_result_code(g("result_code"), g("result_label"))
        entry = {
            "x_rd":            _safe_float(g("x")),
            "y_rd":            _safe_float(g("y")),
            "location_type":   _classify_on_offshore(g("onshore")),
            "hc_result":       hc_result,
            "hc_discovery":    _hc_discovery(hc_result),
            "result_code_raw": (str(g("result_code")).strip()
                                if pd.notna(g("result_code")) else None),
            "field_name":      (str(g("field")).strip()
                                if pd.notna(g("field")) else None),
            "objective":       (str(g("objective")).strip()
                                if pd.notna(g("objective")) else None),
        }
        wells_meta[bh] = entry

    # alias keys — folder names on disk often use different case/separators
    aliased = dict(wells_meta)
    for k, v in wells_meta.items():
        for variant in {k.upper(), k.lower(),
                        k.replace("-", "_"), k.replace("_", "-"),
                        k.replace(" ", "_"), k.replace(" ", "")}:
            if variant and variant not in aliased:
                aliased[variant] = v
    print(f"  indexed {len(wells_meta)} unique wells "
          f"({len(aliased)} with aliases)")
    return aliased


def _lookup_well_meta(folder_name: str, wells_meta: dict) -> dict | None:
    if not wells_meta:
        return None
    for variant in (folder_name, folder_name.upper(), folder_name.lower(),
                    folder_name.replace("-", "_"),
                    folder_name.replace("_", "-")):
        if variant in wells_meta:
            return wells_meta[variant]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LILY
# ─────────────────────────────────────────────────────────────────────────────

def pull_lily(report: PullReport) -> pd.DataFrame:
    print("Pulling LILY ...")
    frames: list[pd.DataFrame] = []

    specs = [
        ("rhob",     "MAD_DataLITH.csv",   "Bulk density (g/cm^3)",
         "Depth CSF-A (m)",     1.0,  4.0),
        ("ngr_cps",  "NGR_DataLITH.csv",   "NGR total counts (cps)",
         "Depth CSF-A (m)",     0,    500),
        ("vp_m_s",   "PWC_DataLITH.csv",   "P-wave velocity x (m/s)",
         "Depth CSF-A (m)",     1000, 7500),
        ("msus_si",  "KAPPA_DataLITH.csv", "Mean susceptibility (SI)",
         "Top depth CSF-A (m)", 1e-8, 1.0),
    ]

    for meas, fname, src_col, depth_col, lo, hi in specs:
        path = LILY_DIR / fname
        if not path.exists():
            print(f"  [skip] {fname} not found")
            continue

        df = pd.read_csv(path, low_memory=False)
        n_in = len(df)
        report.lily_rows_in += n_in

        need = ["Exp", "Site", "Hole", depth_col, "Principal", src_col]
        missing = [c for c in need if c not in df.columns]
        if missing:
            print(f"  [skip] {fname} missing columns {missing}")
            continue

        df = df[need].rename(columns={
            depth_col: "depth",
            "Principal": "lith_principal",
            src_col: "value",
        })
        df["borehole"] = (df["Exp"].astype(str) + "-" +
                          df["Site"].astype(str) + "-" +
                          df["Hole"].astype(str))
        df = df.drop(columns=["Exp", "Site", "Hole"])
        df["measurement"]    = meas
        df["dataset"]        = "LILY"
        df["formation"]      = pd.NA
        df["strat_unit"]     = pd.NA
        df["period"]         = pd.NA
        df["era"]            = pd.NA
        df["x_rd"]           = np.nan
        df["y_rd"]           = np.nan
        df["location_type"]  = pd.NA
        df["hc_result"]      = pd.NA
        df["hc_discovery"]   = pd.NA
        df["field_name"]     = pd.NA

        before = len(df); df = df.dropna(subset=["depth", "value"])
        report.lily_rows_dropped[f"{meas}: missing"] += before - len(df)

        nonfinite = ~np.isfinite(df["value"].values)
        report.lily_nonfinite += int(nonfinite.sum())
        df = df[~nonfinite]

        before = len(df); df = df[(df["value"] >= lo) & (df["value"] <= hi)]
        report.lily_rows_dropped[f"{meas}: out of range [{lo},{hi}]"] += before - len(df)

        before = len(df); df = df[(df["depth"] >= 0) & (df["depth"] < 5000)]
        report.lily_rows_dropped[f"{meas}: depth out of range"] += before - len(df)

        df["rock_type"]      = df["lith_principal"].apply(classify_lithology)
        df["rock_type_fine"] = df["rock_type"]

        frames.append(df[["dataset", "borehole", "depth", "measurement", "value",
                          "rock_type", "rock_type_fine",
                          "formation", "strat_unit", "period", "era",
                          "lith_principal",
                          "x_rd", "y_rd", "location_type",
                          "hc_result", "hc_discovery", "field_name"]])
        print(f"  {meas:9s} : {len(df):>9,} rows (from {n_in:,})")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    mask = out["measurement"] == "vp_m_s"
    out.loc[mask, "value"]       = 304_800.0 / out.loc[mask, "value"]
    out.loc[mask, "measurement"] = "dt_us_ft"
    report.lily_rows_out = len(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NLOG helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _read_strat(folder: Path, report: PullReport) -> list[dict]:
    f = folder / "strat.json"
    if not f.exists() or f.stat().st_size <= 4:
        return []
    try:
        data = json.loads(f.read_text())
    except Exception:
        report.nlog_corrupt_json.append(folder.name)
        return []
    if not data:
        return []
    preferred, any_iv = [], []
    for interp in data.get("stratIntprts", []) or []:
        ivs = interp.get("stratIntvals", []) or []
        any_iv.extend(ivs)
        if interp.get("preferredBln") == "J":
            preferred.extend(ivs)
    return preferred if preferred else any_iv


def _read_details(folder: Path, report: PullReport) -> dict:
    """Fallback: per-well location from details.json if not in boreholes.xlsx."""
    f = folder / "details.json"
    out = {"x_rd": np.nan, "y_rd": np.nan, "location_type": None}
    if not f.exists() or f.stat().st_size <= 4:
        return out
    try:
        data = json.loads(f.read_text())
    except Exception:
        report.details_corrupt.append(folder.name)
        return out
    if not data:
        return out

    for xk in ("xCoord", "x", "easting", "xRd"):
        if xk in data and data[xk] is not None:
            try:
                out["x_rd"] = float(data[xk])
                break
            except (TypeError, ValueError):
                pass
    for yk in ("yCoord", "y", "northing", "yRd"):
        if yk in data and data[yk] is not None:
            try:
                out["y_rd"] = float(data[yk])
                break
            except (TypeError, ValueError):
                pass

    onsh = data.get("onshore")
    if onsh is True or (isinstance(onsh, str) and onsh.lower().startswith("j")):
        out["location_type"] = "onshore"
    elif onsh is False or (isinstance(onsh, str) and onsh.lower().startswith("n")):
        out["location_type"] = "offshore"
    else:
        loc = (data.get("location") or data.get("blok") or "")
        if isinstance(loc, str):
            loc_u = loc.upper()
            if len(loc_u) >= 2 and loc_u[0].isalpha() and loc_u[1].isdigit():
                out["location_type"] = "offshore"
            elif loc_u:
                out["location_type"] = "onshore"
    return out


def _validate_intervals(intervals: list[dict], report: PullReport) -> list[dict]:
    good = []
    for iv in intervals:
        t, b = iv.get("topDepth"), iv.get("bottomDepth")
        if t is None or b is None or t >= b:
            report.strat_bad_intervals += 1
            continue
        good.append(iv)
    sorted_good = sorted(good, key=lambda x: x["topDepth"])
    for prev, curr in zip(sorted_good, sorted_good[1:]):
        if curr["topDepth"] < prev["bottomDepth"]:
            report.strat_overlaps += 1
    return good


def _depth_unit_is_feet(las: lasio.LASFile) -> bool:
    try:
        unit = str(las.well.STRT.unit).upper().strip()
    except Exception:
        return False
    return unit in ("F", "FT", "FEET")


def _validate_las_df(las: lasio.LASFile, df: pd.DataFrame, borehole: str,
                     report: PullReport) -> pd.DataFrame | None:
    df.columns = [c.upper() for c in df.columns]
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df[df.index.notna()]
    if len(df) < 10:
        report.nlog_bad_index.append(borehole)
        return None
    if _depth_unit_is_feet(las):
        df.index = df.index * FT_TO_M
        report.nlog_feet_wells.append(borehole)
    diffs = np.diff(df.index.values)
    consistent = (np.sum(diffs > 0) >= 0.95 * len(diffs) or
                  np.sum(diffs < 0) >= 0.95 * len(diffs))
    if not consistent:
        report.nlog_bad_index.append(borehole)
        return None
    report.large_depth_gaps += int((np.abs(diffs) > MAX_EXPECTED_GAP_M).sum())
    if "RHOB" in df.columns:
        vals = df["RHOB"].dropna()
        if len(vals) > 100 and vals.median() > 100:
            report.nlog_unit_issues.append(borehole)
            df["RHOB"] = df["RHOB"] / 1000.0
    return df


def _flag_gradient_spikes(values: np.ndarray, depths: np.ndarray,
                          threshold_per_m: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)
    dv = np.abs(np.diff(values))
    dd = np.abs(np.diff(depths)).clip(min=1e-6)
    grad = dv / dd
    mask = np.zeros(len(values), dtype=bool)
    mask[1:] = grad > threshold_per_m
    return mask


def _flag_flat_intervals(values: np.ndarray, min_run: int) -> np.ndarray:
    if len(values) < min_run:
        return np.zeros(len(values), dtype=bool)
    mask = np.zeros(len(values), dtype=bool)
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[start]:
            if i - start >= min_run:
                mask[start:i] = True
            start = i
    if len(values) - start >= min_run:
        mask[start:] = True
    return mask


def _resolve_alias(df: pd.DataFrame, aliases: tuple) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _find_formation(strat_unit_id: str, report: PullReport
                    ) -> tuple[str | None, str | None]:
    if not isinstance(strat_unit_id, str) or not strat_unit_id:
        return None, None
    for f in NLOG_FORMATIONS:
        if strat_unit_id.startswith(f):
            return f, strat_unit_id
    report.strat_unknown_prefix[strat_unit_id[:4]] += 1
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# NLOG loop
# ─────────────────────────────────────────────────────────────────────────────

def pull_nlog(
    report: PullReport,
    wells_meta: dict | None = None,
    max_wells: int | None = None,
) -> pd.DataFrame:
    print("\nPulling NLOG (walks every LAS) ...")
    wells_meta = wells_meta or {}
    folders = sorted(f for f in NLOG_DIR.iterdir() if f.is_dir())
    if max_wells:
        folders = folders[:max_wells]
    report.nlog_wells_in = len(folders)

    rows = []
    for i, folder in enumerate(folders, 1):
        if i % 200 == 0:
            print(f"  NLOG: {i}/{len(folders)}   rows so far = {len(rows):,}")

        intervals = _read_strat(folder, report)
        if intervals:
            report.nlog_wells_with_strat += 1
            intervals = _validate_intervals(intervals, report)
        if not intervals:
            report.nlog_empty_strat.append(folder.name)
            continue

        las_files = list(folder.glob("*.las"))
        if not las_files:
            continue
        report.nlog_wells_with_las += 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                las = lasio.read(str(las_files[0]), ignore_header_errors=True)
            df = las.df()
        except Exception:
            report.nlog_bad_index.append(folder.name)
            continue

        df = _validate_las_df(las, df, folder.name, report)
        if df is None:
            continue

        for c in df.columns:
            report.nlog_curve_counts[c] += 1

        if set(df.columns) & SONIC_SHEAR_MNEMONICS:
            report.nlog_shear_wells.append(folder.name)

        # location + hc_result metadata
        # primary source: boreholes.xlsx; fallback: details.json (no hc_result)
        meta = _lookup_well_meta(folder.name, wells_meta)
        if meta is not None:
            report.xlsx_matched += 1
        else:
            report.xlsx_unmatched.append(folder.name)
            details = _read_details(folder, report)
            meta = {
                "x_rd":          details["x_rd"],
                "y_rd":          details["y_rd"],
                "location_type": details["location_type"],
                "hc_result":     None,
                "hc_discovery":  None,
                "field_name":    None,
            }

        if np.isnan(meta["x_rd"]) or np.isnan(meta["y_rd"]):
            report.coords_missing += 1
        if meta["location_type"]:
            report.location_hits[meta["location_type"]] += 1
        if meta.get("hc_result"):
            report.result_hits[meta["hc_result"]] += 1
        disc = meta.get("hc_discovery")
        if disc is True:
            report.discovery_hits["positive"] += 1
        elif disc is False:
            report.discovery_hits["negative"] += 1
        else:
            report.discovery_hits["unknown"] += 1

        resolved: list[tuple[str, str, CurveSpec]] = []
        for out_name, spec in ALL_NLOG_SPECS.items():
            las_col = _resolve_alias(df, spec.aliases)
            if las_col is None:
                continue
            resolved.append((out_name, las_col, spec))
            report.alias_hits[(out_name, las_col)] += 1

        if not resolved:
            continue

        depths = df.index.values
        depth_to_formation  = np.full(len(depths), None, dtype=object)
        depth_to_strat_unit = np.full(len(depths), None, dtype=object)
        for iv in intervals:
            formation, strat_unit = _find_formation(
                iv.get("stratUnitId", ""), report)
            if formation is None:
                continue
            top, base = iv["topDepth"], iv["bottomDepth"]
            mask = (depths >= top) & (depths <= base)
            if mask.any():
                depth_to_formation[mask]  = formation
                depth_to_strat_unit[mask] = strat_unit
                report.formation_hits[formation] += 1

        step_samples = 1
        if len(depths) > 1:
            dz = float(np.median(np.diff(depths[:1000])))
            if dz != 0:
                step_samples = max(1, int(round(DEPTH_STEP_M / abs(dz))))
        sample_idx = np.arange(0, len(depths), step_samples)

        curve_flags: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for out_name, las_col, spec in resolved:
            vals = df[las_col].values.astype(float)
            spike_mask = (_flag_gradient_spikes(vals, depths, spec.gradient_lim)
                          if spec.gradient_lim is not None
                          else np.zeros(len(vals), dtype=bool))
            flat_mask = _flag_flat_intervals(vals, FLAT_INTERVAL_MIN_SAMPLES)
            curve_flags[out_name] = (spike_mask, flat_mask)
            report.gradient_spikes[out_name] += int(spike_mask.sum())
            report.flat_intervals[out_name]  += int(flat_mask.sum())

        used_any = False
        for idx in sample_idx:
            formation = depth_to_formation[idx]
            if formation is None:
                continue
            strat_unit = depth_to_strat_unit[idx]
            depth = float(depths[idx])

            rock_type = NLOG_FORMATION_TO_ROCK.get(formation, "other")
            rock_type_fine = classify_strat_unit(strat_unit, formation)
            # flag Zechstein intervals with only group-level codes (ZEZ1,
            # ZEZ2, ...) — these have ambiguous lithology and are dropped
            # to 'other' by the updated classifier
            if formation == "ZE" and rock_type_fine == "other" and strat_unit:
                report.ze_group_only[strat_unit] += 1
            if strat_unit and strat_unit not in STRAT_UNIT_TO_FINE_ROCK:
                matched = any(strat_unit.startswith(p)
                              for p in STRAT_UNIT_TO_FINE_ROCK)
                if matched:
                    report.strat_unit_hits[strat_unit] += 1
                else:
                    report.strat_unit_unknown[strat_unit] += 1
            elif strat_unit:
                report.strat_unit_hits[strat_unit] += 1

            period = NLOG_FORMATION_TO_PERIOD.get(formation)
            era    = PERIOD_TO_ERA.get(period) if period else None

            for out_name, las_col, spec in resolved:
                raw = df[las_col].iloc[idx]
                if pd.isna(raw) or not np.isfinite(raw):
                    report.nonfinite_in_las += 1
                    continue

                spike_mask, flat_mask = curve_flags[out_name]
                if spike_mask[idx] or flat_mask[idx]:
                    continue

                if spec.transform is not None:
                    try:
                        val = spec.transform(float(raw))
                    except Exception:
                        continue
                    if not np.isfinite(val):
                        continue
                else:
                    val = float(raw)

                lo, hi = spec.bounds
                if val < lo:
                    report.clipping_counts[(out_name, "below")] += 1
                    continue
                if val > hi:
                    report.clipping_counts[(out_name, "above")] += 1
                    continue
                if out_name == "dt_us_ft" and depth >= 500 and val > 130:
                    report.clipping_counts[("dt_us_ft", "shear-suspect")] += 1
                    continue

                rows.append({
                    "dataset":        "NLOG",
                    "borehole":       folder.name,
                    "depth":          depth,
                    "measurement":    out_name,
                    "value":          val,
                    "rock_type":      rock_type,
                    "rock_type_fine": rock_type_fine,
                    "formation":      formation,
                    "strat_unit":     strat_unit,
                    "period":         period,
                    "era":            era,
                    "lith_principal": pd.NA,
                    "x_rd":           meta["x_rd"],
                    "y_rd":           meta["y_rd"],
                    "location_type":  meta["location_type"],
                    "hc_result":      meta.get("hc_result"),
                    "hc_discovery":   meta.get("hc_discovery"),
                    "field_name":     meta.get("field_name"),
                })
                report.per_feature_rows[out_name] += 1
                report.per_feature_wells[out_name].add(folder.name)
                used_any = True

        if used_any:
            report.nlog_wells_used += 1

    df_out = pd.DataFrame(rows)
    report.nlog_rows_out = len(df_out)
    return df_out


def tier2_checks(df: pd.DataFrame, report: PullReport) -> pd.DataFrame:
    if len(df) == 0:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=["dataset", "borehole", "depth", "measurement"],
                            keep="first")
    report.duplicate_rows += before - len(df)

    for (ds, meas), grp in df.groupby(["dataset", "measurement"]):
        well_medians = grp.groupby("borehole")["value"].median()
        if len(well_medians) < 5:
            continue
        pop_median = well_medians.median()
        mad = (well_medians - pop_median).abs().median()
        if mad == 0:
            continue
        outliers = well_medians[(well_medians - pop_median).abs()
                                > OUTLIER_MAD_MULTIPLIER * mad]
        if len(outliers):
            report.outlier_wells[f"{ds}/{meas}"] = outliers.index.tolist()

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--boreholes", type=Path, default=NLOG_BOREHOLES_XLSX,
                    help="path to NLOG boreholes.xlsx (location + result metadata)")
    ap.add_argument("--nlog-max", type=int, default=None)
    ap.add_argument("--skip-lily", action="store_true")
    ap.add_argument("--skip-nlog", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = PullReport()
    frames = []

    wells_meta: dict = {}
    if not args.skip_nlog:
        print("\nLoading boreholes.xlsx metadata ...")
        wells_meta = _load_boreholes_metadata(args.boreholes, report)

    if not args.skip_lily:
        frames.append(pull_lily(report))
    if not args.skip_nlog:
        frames.append(pull_nlog(report, wells_meta=wells_meta,
                                max_wells=args.nlog_max))

    frames = [f for f in frames if len(f)]
    if not frames:
        print("no data produced")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTier-2 sanity checks on {len(df):,} pooled rows ...")
    df = tier2_checks(df, report)

    for c in ("formation", "strat_unit", "period", "era",
              "lith_principal", "location_type",
              "rock_type", "rock_type_fine", "hc_result", "field_name"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    out_path = args.out / "samples.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n→ wrote {out_path}  ({len(df):,} rows, "
          f"{out_path.stat().st_size/1e6:.1f} MB)")

    report.write(args.out / "pull_report.txt")
    print(f"→ done. outputs in {args.out}/")


if __name__ == "__main__":
    main()