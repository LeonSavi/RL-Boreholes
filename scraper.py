"""
Download: composite LAS files, details.json, strat.json, dirsurvey.json

All four land in data/nlog/nlog_scrape/<Borehole code>/, and one row per
borehole is appended to _scrape_log.csv.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

BASE = "https://www.nlog.nl"
WELLS_QUERY = f"{BASE}/standalone/rest/services/nlog_gdn/gdw_ng_wll_all_utm_v1/MapServer/0/query"
MV = f"{BASE}/nlog-mapviewer/rest/brh"
DOC_DL = f"{BASE}/brh-web/rest/brh/logdocument"

HEADERS = {
    "User-Agent": "RL-Boreholes-research/0.1 (academic thesis)",
    "Accept": "application/json",
}

DEFAULT_XLSX = Path("data/nlog/boreholes.xlsx")
DEFAULT_OUT = Path("data/nlog/nlog_scrape")


# --------------------------------------------------------------------------
# Thread-local sessions: each worker thread gets its own keep-alive pool.
# --------------------------------------------------------------------------
_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        _tls.session = s
    return s


_log_lock = threading.Lock()


# ==========================================================================
# Step 1 — SHORT_NM -> BOREHOLE_DBK (bulk, sequential)
# ==========================================================================
def _sql_quote(name: str) -> str:
    return name.replace("'", "''")


def resolve_codes_bulk(codes: list[str], batch_size: int = 200) -> dict[str, list[dict]]:
    s = _session()
    out: dict[str, list[dict]] = {c: [] for c in codes}
    for i in range(0, len(codes), batch_size):
        chunk = codes[i:i + batch_size]
        in_list = ",".join(f"'{_sql_quote(c)}'" for c in chunk)
        r = s.get(
            WELLS_QUERY,
            params={
                "where": f"SHORT_NM IN ({in_list})",
                "outFields": "BOREHOLE_DBK,SHORT_NM",
                "returnGeometry": "false",
                "f": "json",
            },
            timeout=60,
        )
        r.raise_for_status()
        for feat in r.json().get("features", []):
            a = feat["attributes"]
            nm = a.get("SHORT_NM")
            if nm in out:
                out[nm].append(a)
        print(f"  resolved {i + len(chunk):>5}/{len(codes)}")
        time.sleep(0.1)
    return out


# ==========================================================================
# Per-borehole REST endpoints
# ==========================================================================
def _post_json(endpoint: str, dbk: int | str):
    s = _session()
    # /details wants a LIST; the others want a bare number
    if endpoint.endswith("/details"):
        r = s.post(endpoint, json=[str(int(dbk))],
                   headers={"Content-Type": "application/json"}, timeout=30)
    else:
        r = s.post(endpoint, data=str(int(dbk)),
                   headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    if r.status_code == 204 or not r.text.strip():
        return None
    return r.json()


def list_log_documents(dbk: int) -> list[dict]:
    return _post_json(f"{MV}/logdocuments", dbk) or []


def get_details(dbk: int):
    data = _post_json(f"{MV}/details", dbk)
    if isinstance(data, list) and data:
        return data[0]
    return data


def get_strat(dbk: int):
    return _post_json(f"{MV}/stratinterpretations", dbk)


def get_dirsurvey(dbk: int):
    return _post_json(f"{MV}/dirsurveys", dbk)


def download_document(doc_dbk: int, out_path: Path, chunk: int = 1 << 16) -> Path:
    s = _session()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with s.get(f"{DOC_DL}/{int(doc_dbk)}", stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for part in r.iter_content(chunk_size=chunk):
                fh.write(part)
    return out_path


# ============================
# LOGS
# ====================================================
LOG_FIELDS = [
    "code", "borehole_dbk",
    "n_las", "bytes_las",
    "has_details", "has_strat", "has_dirsurvey",
    "n_strat_intervals",
    "status",      
    "error",
]


def _read_log(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return {row["code"]: row for row in csv.DictReader(fh)}


def _append_log(path: Path, row: dict) -> None:
    with _log_lock:
        new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)



def _las_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() == ".las"]


def _json_present(folder: Path, fname: str) -> bool:
    p = folder / fname
    # "null" files (4 bytes incl. newline) are valid "we asked, got 204" markers;
    # but we only consider an actual JSON object/array a hit.
    return p.is_file() and p.stat().st_size > 4


def _folder_complete(folder: Path) -> bool:
    """A borehole folder is 'done' when it has at least one LAS and all three
    JSON metadata files are present (any, including 'null' markers)."""
    if not folder.is_dir():
        return False
    has_las  = bool(_las_files(folder))
    has_det  = (folder / "details.json").is_file()
    has_str  = (folder / "strat.json").is_file()
    has_dir  = (folder / "dirsurvey.json").is_file()
    return has_las and has_det and has_str and has_dir


# ==========================================================================
# Worker
# ==============================================================
FileFilter = Literal["composite_las", "all_las", "all_logs"]


def _keep(doc: dict, mode: FileFilter) -> bool:
    ft = doc.get("fileTypeCode")
    grp = doc.get("documentGroup")
    if mode == "composite_las":
        return ft == "LAS" and grp == "COMPOSITE_LOG_FILE"
    if mode == "all_las":
        return ft == "LAS"
    return True


def _process_one(
    code: str,
    dbk: int | None,
    out_dir: Path,
    mode: FileFilter,
    sleep: float,
) -> dict:
    """Fetch LAS + details + strat + dirsurvey for one borehole."""
    row: dict = {
        "code": code, "borehole_dbk": dbk or "",
        "n_las": 0, "bytes_las": 0,
        "has_details": False, "has_strat": False, "has_dirsurvey": False,
        "n_strat_intervals": 0,
        "status": "", "error": "",
    }

    if dbk is None:
        row["status"] = "not_found"
        row["error"]  = "no ArcGIS match"
        return row

    bh_dir = out_dir / code
    bh_dir.mkdir(parents=True, exist_ok=True)
    errs: list[str] = []

    # ------ LAS files ---------------------------------------------------
    try:
        docs = list_log_documents(dbk)
    except Exception as e:
        errs.append(f"logdocuments:{e!r}"[:150])
        docs = []

    wanted = [d for d in docs if _keep(d, mode)]
    for d in wanted:
        dst = bh_dir / d["fileName"]
        expected = int(d.get("fileSize") or 0)
        if dst.exists() and (expected == 0 or dst.stat().st_size == expected):
            row["n_las"] += 1
            row["bytes_las"] += dst.stat().st_size
            continue
        try:
            download_document(d["documentBfileDbk"], dst)
            row["n_las"] += 1
            row["bytes_las"] += dst.stat().st_size
            if sleep:
                time.sleep(sleep)
        except Exception as e:
            errs.append(f"las({d['fileName']}):{e!r}"[:150])
            break

    # ------ JSON metadata ----------------------------------------------
    json_calls = [
        ("details.json",   get_details,   "has_details"),
        ("strat.json",     get_strat,     "has_strat"),
        ("dirsurvey.json", get_dirsurvey, "has_dirsurvey"),
    ]
    for fname, fn, flag in json_calls:
        dst = bh_dir / fname
        if dst.is_file() and dst.stat().st_size > 4:
            row[flag] = True
            continue
        try:
            data = fn(dbk)
            if data is None:
                dst.write_text("null")                    # 204-style marker
            else:
                dst.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                row[flag] = True
            if sleep:
                time.sleep(sleep)
        except Exception as e:
            errs.append(f"{fname}:{e!r}"[:150])

    # ------ strat interval count (cheap to extract, very useful later) --
    if row["has_strat"]:
        try:
            strat = json.loads((bh_dir / "strat.json").read_text())
            total = 0
            for ip in strat.get("stratIntprts", []) or []:
                total += len(ip.get("intervals", []) or [])
            row["n_strat_intervals"] = total
        except Exception:
            pass

    # ------ status summary ---------------------------------------------
    if row["n_las"] == 0 and len(wanted) == 0:
        row["status"] = "no_las"     # borehole has no composite LAS attached
    elif row["n_las"] == len(wanted) and not errs:
        row["status"] = "success"
    else:
        row["status"] = "partial"
    if errs:
        row["error"] = " | ".join(errs)
    return row


# ==========================================================================
# Driver
# ==========================================================================
def run(
    xlsx_path: Path = DEFAULT_XLSX,
    out_dir: Path = DEFAULT_OUT,
    mode: FileFilter = "composite_las",
    limit: int | None = None,
    filter_onshore: bool | None = None,
    code_col: str = "Borehole code",
    max_workers: int = 8,
    sleep_per_request: float = 0.0,
) -> None:
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_scrape_log.csv"

    # --- pick codes -----------------------------------------------------
    df = pd.read_excel(xlsx_path)
    if filter_onshore is True:
        df = df[df["On offshore"] == "ON"]
    elif filter_onshore is False:
        df = df[df["On offshore"] == "OFF"]
    codes = df[code_col].dropna().astype(str).unique().tolist()
    if limit:
        codes = codes[:limit]
    print(f"[init] {len(codes)} boreholes in scope (mode={mode}, out={out_dir})")

    # --- filesystem-grounded resume ------------------------------------
    # A borehole is "done" if its folder has LAS + all 3 JSON files.
    # Boreholes previously logged as no_las / not_found are also skipped,
    # but only if the log row exists (so manual cleanup re-queues them).
    log_rows = _read_log(log_path)
    skip_terminal = {"no_las", "not_found"}

    def _is_done(code: str) -> bool:
        if _folder_complete(out_dir / code):
            return True
        r = log_rows.get(code)
        return bool(r and r.get("status") in skip_terminal)

    remaining = [c for c in codes if not _is_done(c)]
    print(f"[init] {len(codes) - len(remaining)} already done, "
          f"{len(remaining)} to fetch")
    if not remaining:
        return

    # --- SHORT_NM -> BOREHOLE_DBK --------------------------------------
    print(f"[step 1] resolving {len(remaining)} codes to BOREHOLE_DBK...")
    resolved = resolve_codes_bulk(remaining)
    dbk_of = {c: (int(v[0]["BOREHOLE_DBK"]) if v else None) for c, v in resolved.items()}
    n_found = sum(1 for v in dbk_of.values() if v is not None)
    print(f"[step 1] matched {n_found}/{len(remaining)}")

    # --- fan out --------------------------------------------------------
    print(f"[step 2] fetching with {max_workers} workers "
          f"(LAS + details + strat + dirsurvey per borehole)...")
    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, c, dbk_of[c], out_dir, mode, sleep_per_request): c
            for c in remaining
        }
        for fut in as_completed(futures):
            row = fut.result()
            _append_log(log_path, row)
            done_count += 1
            # one-line summary, LAS count + JSON flags
            flags = "".join(
                k[0].upper() if row[f"has_{k}"] else "-"
                for k in ("details", "strat", "dirsurvey")
            )
            if row["status"] in {"success", "partial"} or row["error"]:
                print(f"  [{done_count:>5}/{len(remaining)}] {row['code']:<14} "
                      f"{row['status']:8} LAS={row['n_las']} "
                      f"({row['bytes_las']/1e6:.2f} MB) json={flags} "
                      f"strat_intervals={row['n_strat_intervals']}"
                      + (f"  ERR: {row['error']}" if row["error"] else ""))
            elif done_count % 50 == 0:
                print(f"  [{done_count:>5}/{len(remaining)}] progress")


# ==========================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--mode", default="composite_las",
                   choices=["composite_las", "all_las", "all_logs"])
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N boreholes (pilot).")
    p.add_argument("--workers", type=int, default=20,
                   help="Concurrent workers (default 20).")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Per-request sleep inside each worker (seconds).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--onshore", action="store_true")
    g.add_argument("--offshore", action="store_true")
    args = p.parse_args()

    run(
        xlsx_path=Path(args.xlsx),
        out_dir=Path(args.out),
        mode=args.mode,
        limit=args.limit,
        filter_onshore=True if args.onshore else False if args.offshore else None,
        max_workers=args.workers,
        sleep_per_request=args.sleep,
    )