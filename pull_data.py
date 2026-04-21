"""
Pull LILY + NLOG into ONE unified Parquet (`samples.parquet`) with a
`dataset` column. Expanded feature set + all Tier 1+2 sanity checks.

Output schema
-------------
dataset          str       'LILY' | 'NLOG'
borehole         str       well/site identifier
depth            float     metres below deposition surface
measurement      str       rhob | gr_api | dt_us_ft | ...
value            float     cleaned measurement
rock_type        str       coarse rock-type (unchanged from v3)
rock_type_fine   str       sub-formation-aware rock-type (NEW)
formation        string    NLOG formation code (e.g. 'RO') or <NA>
strat_unit       string    NLOG sub-unit code (e.g. 'ROSLV') or <NA>
period           string    e.g. 'Permian' or <NA>
era              string    'Cenozoic' | 'Mesozoic' | 'Paleozoic' or <NA>
lith_principal   string    LILY principal lithology or <NA>
x_rd             float     NLOG x (Dutch RD) or NaN
y_rd             float     NLOG y (Dutch RD) or NaN
location_type    string    'onshore' | 'offshore' | <NA>

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


LILY_DIR    = Path("data/lily")
NLOG_DIR    = Path("data/nlog/nlog_scrape")
DEFAULT_OUT = Path("data/clean")

DEPTH_STEP_M = 1.0
FT_TO_M      = 0.3048 # LS: some depath in nlog are in feet 

NLOG_FORMATIONS = ["NU", "CK", "KN", "RN", "RB", "ZE", "RO",
                   "NM", "NL", "DC", "AT", "SL", "SG", "SK"]

# Coarse rock-type mapping per formation
NLOG_FORMATION_TO_ROCK = {
    "NU": "clay",       "CK": "chalk",     "KN": "claystone",
    "RN": "claystone",  "RB": "sandstone", "ZE": "halite",
    "RO": "sandstone",  "NM": "clay",      "NL": "clay",
    "DC": "claystone",  "AT": "claystone", "SL": "claystone",
    "SG": "claystone",  "SK": "claystone",
}

# Geologic era / period per formation prefix
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

# Sub-unit/ stratUnitId: fine lithology. refined by claude.
# These are the Dutch sub-members we most care about; unknown codes fall
# back to the coarse NLOG_FORMATION_TO_ROCK assignment. Keys are prefixes
# of the full stratUnitId so we match with startswith().
# Medium-coverage dictionary (~40 entries).
STRAT_UNIT_TO_FINE_ROCK = {
    # ─── Rotliegend (RO) ─── ~270 Ma, continental
    "ROSL":  "sandstone",          # Slochteren Formation (reservoir sand)
    "ROSLU": "sandstone",          # Upper Slochteren Member
    "ROSLV": "sandstone",          # Lower Slochteren / Volpriehausen
    "ROCLT": "claystone",          # Ten Boer Claystone Member
    "ROCL":  "claystone",          # other Rotliegend claystones
    "ROSS":  "sandstone",          # Silverpit Formation — sandier parts
    "ROSSF": "claystone",          # Silverpit fine-grained
    # ─── Zechstein (ZE) ─── ~255 Ma, evaporites
    "ZEZ1":  "halite",             # Z1 (Werra) — mixed, dominantly halite
    "ZEZ1C": "carbonate",          # Z1 Carbonate
    "ZEZ1A": "anhydrite",          # Z1 Anhydrite
    "ZEZ1H": "halite",             # Z1 Halite
    "ZEZ2":  "halite",             # Z2 (Stassfurt) — dominantly halite
    "ZEZ2C": "carbonate",          # Z2 Carbonate (Hauptdolomit)
    "ZEZ2A": "anhydrite",          # Z2 Basal Anhydrite
    "ZEZ2H": "halite",             # Z2 Halite
    "ZEZ3":  "halite",             # Z3 (Leine) — dominantly halite
    "ZEZ3C": "carbonate",
    "ZEZ3A": "anhydrite",
    "ZEZ3H": "halite",
    "ZEZ4":  "halite",             # Z4 (Aller)
    "ZEZ4A": "anhydrite",
    "ZEZ4H": "halite",
    "ZEZ5":  "halite",             # Z5 (Ohre)
    "ZESA":  "anhydrite",   # 63k rows
    "ZESAU": "anhydrite",   # 14k rows (upper)
    "ZESAL": "anhydrite",   # 25k rows (lower)
    # ─── Buntsandstein (RB) ─── ~245 Ma, continental sand with mud
    "RBM":   "sandstone",          # Main Buntsandstein
    "RBMH":  "sandstone",          # Hardegsen
    "RBMV":  "sandstone",          # Volpriehausen (within Bunt)
    "RBMD":  "sandstone",          # Detfurth
    "RBSH":  "claystone",          # Solling Claystone
    "RBSHS": "claystone",          # Solling Shale
    # ─── Muschelkalk / Keuper (RN) ─── ~235 Ma
    "RNRO":  "claystone",          # Röt Formation (mudstone + anhydrite)
    "RNROC": "carbonate",          # Röt Carbonate
    "RNROE": "anhydrite",          # Röt Evaporite
    "RNMU":  "carbonate",          # Muschelkalk (limestone/dolomite)
    "RNMUE": "anhydrite",
    "RNMUC": "carbonate",
    "RNKPU": "claystone",          # Upper Keuper
    "RNKPL": "claystone",          # Lower Keuper
    # ─── Chalk (CK) ─── ~100 Ma
    "CKEK":  "chalk",              # Ekofisk
    "CKTX":  "chalk",              # Texel
    "CKGR":  "chalk",              # Ommelanden / other
    # ─── Rijnland (KN) ─── ~130 Ma
    "KNNC":  "claystone",
    "KNNS":  "sandstone",          # Rijnland sandier intervals
    "KNGL":  "claystone",

    # Altena Group — Lower Jurassic marine mudstone
    "ATAL":  "claystone",   # 63k rows — Aalburg Fm, marine shale
    "ATWDL": "claystone",   # 8.5k rows — Werkendam, Lower
    # ─── Cenozoic catch-alls ───
    "NUBA":  "clay",               # Breda
    "NUIE":  "clay",               # IJsselmeer
    "NLFF":  "clay",               # Lower North Sea
    "NMDO":  "clay",               # Middle North Sea / Dongen
    
    # Schieland (Upper Jurassic)
    "SLDNA": "claystone",   # 28k rows — Delfland Formation

    # Cenozoic / Paleogene finer splits
    "NUOT":  "clay",        # 55k rows — Oosterhout Fm
    "NUMS":  "clay",        # 23k rows — Middle North Sea
    "NLLFC": "clay",        # 20k rows — Landen Formation clay
    "NMRF":  "clay",        # 14k rows — Rupel Formation
    "NMRFC": "clay",        # 15k rows — Rupel Clay

    # Carboniferous Coal Measures
    "DCCU":  "claystone",   # 15k rows — Upper Carboniferous, coal-bearing
    "DCCR":  "claystone",   # 13k rows — Caumer Subgroup

    # Muschelkalk Solling Carbonate (probably)
    "RNSOC": "carbonate",   # 11k rows

    # additional Jurassic-Triassic shales (all dominantly claystone)
    "ATRT":  "claystone",    # Altena Röt — mixed claystone
    "ATWDU": "claystone",    # Werkendam Upper
    "SGKI":  "claystone",    # Schieland Kimmeridge
    "SLDNR": "claystone",    # Schieland Delfland Rosendahl  
    "SLDND": "claystone",    # Schieland Delfland
    "SLDN":  "claystone",    # Schieland Delfland (generic)
    "SLCL":  "claystone",    # Schieland claystone
    "RNKP":  "claystone",    # Keuper (generic)
    "RNKPS": "claystone",    # Keuper subunit
    "DCDT":  "claystone",    # Dinantian
    "DCDG":  "claystone",    # Dinantian
    "DCHL":  "claystone",    # Hellevoetsluis
    "DCGE":  "claystone",    # Geverik
    "ZEUC":  "carbonate",    # Zechstein Upper Carbonate
}

# ─────────────────────────────────────────────────────────────────────────────
# Curve specifications 
# ─────────────────────────────────────────────────────────────────────────────

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
    """Map a Dutch stratUnitId to a fine-grained rock type, with fallback
    to the coarse formation-level assignment. Uses longest-prefix match."""
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
    clipping_counts:       dict = field(default_factory=lambda: defaultdict(int))
    nonfinite_in_las:      int = 0
    per_feature_rows:      Counter = field(default_factory=Counter)
    per_feature_wells:     dict = field(default_factory=lambda: defaultdict(set))
    alias_hits:            Counter = field(default_factory=Counter)

    location_hits:        Counter = field(default_factory=Counter)
    coords_missing:       int = 0
    details_corrupt:      list = field(default_factory=list)

    duplicate_rows:    int = 0
    outlier_wells:     dict = field(default_factory=lambda: defaultdict(list))
    gradient_spikes:   dict = field(default_factory=lambda: defaultdict(int))
    flat_intervals:    dict = field(default_factory=lambda: defaultdict(int))
    large_depth_gaps:  int = 0

    def write(self, path: Path) -> None:
        L: list = []
        add = L.append
        add("=" * 78)
        add(" PULL + SANITY REPORT  (v4)")
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

        add(f"\n## Location metadata")
        add(f"  coords missing                    {self.coords_missing:>14,}")
        if self.details_corrupt:
            add(f"  corrupt details.json ({len(self.details_corrupt)} wells)")
        for loc_type, n in sorted(self.location_hits.items(), key=lambda x: -x[1]):
            add(f"    {loc_type:<16s} wells={n:>6,}")

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
            add(f"  (review with Charlie — these are where the fine split is missing)")
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
        df["rock_type_fine"] = df["rock_type"]  # LILY has no sub-formation concept

        frames.append(df[["dataset", "borehole", "depth", "measurement", "value",
                          "rock_type", "rock_type_fine",
                          "formation", "strat_unit", "period", "era",
                          "lith_principal",
                          "x_rd", "y_rd", "location_type"]])
        print(f"  {meas:9s} : {len(df):>9,} rows (from {n_in:,})")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    mask = out["measurement"] == "vp_m_s"
    out.loc[mask, "value"]       = 304_800.0 / out.loc[mask, "value"]
    out.loc[mask, "measurement"] = "dt_us_ft"
    report.lily_rows_out = len(out)
    return out



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
    """Extract location metadata from details.json. Returns a dict with
    x_rd, y_rd, location_type (best effort — missing fields are None)."""
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

    # NLOG details.json is a flat dict of well attributes.
    # Coordinate fields: look for xCoord/yCoord, x/y, easting/northing.
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

    # Onshore / offshore: field can be 'onshore' (bool or string) or 'location'
    onsh = data.get("onshore")
    if onsh is True or (isinstance(onsh, str) and onsh.lower().startswith("j")):
        out["location_type"] = "onshore"
    elif onsh is False or (isinstance(onsh, str) and onsh.lower().startswith("n")):
        out["location_type"] = "offshore"
    else:
        # fallback: sometimes there's a 'location' or 'blok' field indicating block
        loc = (data.get("location") or data.get("blok") or "")
        if isinstance(loc, str):
            loc_u = loc.upper()
            # Offshore Dutch blocks are single-letter + digits (A-P, with A-Q covering offshore)
            # Onshore wells typically have text names like 'ROTTERDAM' or 'GRONINGEN'
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
    """Return (formation_2char, full_strat_unit) or (None, None) if the
    strat_unit doesn't match any known formation prefix."""
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

def pull_nlog(report: PullReport, max_wells: int | None = None) -> pd.DataFrame:
    print("\nPulling NLOG (walks every LAS ...")
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

        # location metadata — pulled once per well
        details = _read_details(folder, report)
        if np.isnan(details["x_rd"]) or np.isnan(details["y_rd"]):
            report.coords_missing += 1
        if details["location_type"]:
            report.location_hits[details["location_type"]] += 1

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

        # Map each depth index to (formation, strat_unit)
        depth_to_formation = np.full(len(depths), None, dtype=object)
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

            # Determine rock-type (coarse + fine) for this depth
            rock_type = NLOG_FORMATION_TO_ROCK.get(formation, "other")
            rock_type_fine = classify_strat_unit(strat_unit, formation)
            # Track which sub-units were matched vs unmatched
            if strat_unit and strat_unit not in STRAT_UNIT_TO_FINE_ROCK:
                # Check longest-prefix match the same way classify_strat_unit does
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
                    "x_rd":           details["x_rd"],
                    "y_rd":           details["y_rd"],
                    "location_type":  details["location_type"],
                })
                report.per_feature_rows[out_name] += 1
                report.per_feature_wells[out_name].add(folder.name)
                used_any = True

        if used_any:
            report.nlog_wells_used += 1

    df_out = pd.DataFrame(rows)
    report.nlog_rows_out = len(df_out)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# validate data
# ─────────────────────────────────────────────────────────────────────────────

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


#main 
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nlog-max", type=int, default=None)
    ap.add_argument("--skip-lily", action="store_true")
    ap.add_argument("--skip-nlog", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = PullReport()
    frames = []

    if not args.skip_lily:
        frames.append(pull_lily(report))
    if not args.skip_nlog:
        frames.append(pull_nlog(report, max_wells=args.nlog_max))

    frames = [f for f in frames if len(f)]
    if not frames:
        print("no data produced")
        return

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTier-2 sanity checks on {len(df):,} pooled rows ...")
    df = tier2_checks(df, report)

    for c in ("formation", "strat_unit", "period", "era",
              "lith_principal", "location_type",
              "rock_type", "rock_type_fine"):
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