#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Bio-Logic EC-Lab ASCII export (.txt / .mpt) to Excel (.xlsx)
- Detects correct column header row (handles your file where header is at Nb header lines - 1)
- Optionally handles a separate units row
- Cleans/normalizes column names
- Writes two sheets: data, metadata
"""

import re
import os
import sys
import pandas as pd
from typing import Tuple, List

# ------------------------- helpers -------------------------

def _looks_like_number(s: str) -> bool:
    s = str(s).strip()
    if not s:
        return False
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False

def _parse_nb_header_lines(lines: List[str]) -> int:
    """Return 'Nb header lines' if present, else -1."""
    for raw in lines[:200]:  # search near top
        m = re.match(r"^\s*Nb\s+header\s+lines\s*:\s*(\d+)", raw, flags=re.I)
        if m:
            return int(m.group(1))
    return -1

def _find_header_and_units(lines: List[str], nb_header: int) -> Tuple[int, bool]:
    """
    Determine the *absolute* line for column names (header_row_abs) and whether the next
    line is a units row. For many EC-Lab files (including yours), the actual column names
    are at (Nb header lines - 1), and the first data row starts at (Nb header lines).
    """
    # Strong prior from EC-Lab format:
    if nb_header >= 1 and nb_header < len(lines):
        candidate = nb_header - 1
        row = lines[candidate].rstrip("\n")
        if "\t" in row:
            # Decide if the next line is a units row
            has_units = False
            if candidate + 1 < len(lines):
                nxt = lines[candidate + 1].rstrip("\n")
                if "\t" in nxt:
                    ntok = [t.strip() for t in nxt.split("\t")]
                    score_units = sum(('/' in t) or ('(' in t) or (')' in t) for t in ntok)
                    numericish = sum(_looks_like_number(t) for t in ntok)
                    # units row usually has slashes/parentheses and few pure numbers
                    if score_units >= max(2, len(ntok) // 4) and numericish <= len(ntok) // 3:
                        has_units = True
            return candidate, has_units

    # Fallback: scan downwards from nb_header (or 0 if not present) for a header-ish line
    start = nb_header if nb_header >= 0 else 0
    for abs_row in range(start, min(start + 10, len(lines))):
        r = lines[abs_row].rstrip("\n")
        if "\t" not in r:
            continue
        toks = r.split("\t")
        alphaish = sum(any(ch.isalpha() for ch in t) for t in toks)
        if alphaish >= max(2, len(toks) // 3):  # likely column names
            has_units = False
            if abs_row + 1 < len(lines):
                nxt = lines[abs_row + 1].rstrip("\n")
                if "\t" in nxt:
                    ntok = nxt.split("\t")
                    score_units = sum(('/' in t) or ('(' in t) or (')' in t) for t in ntok)
                    numericish = sum(_looks_like_number(t) for t in ntok)
                    if score_units >= len(ntok) // 3 and numericish <= len(ntok) // 3:
                        has_units = True
            return abs_row, has_units

    # Ultimate fallback: treat nb_header as the header row (no units)
    return max(0, nb_header), False

def _collect_metadata_pairs(lines: List[str], stop_before: int) -> pd.DataFrame:
    """Collect simple 'key: value' pairs from the text header region."""
    kv = []
    for raw in lines[:stop_before]:
        if ":" in raw:
            k, v = raw.split(":", 1)
            kv.append((k.strip(), v.strip()))
    return pd.DataFrame(kv, columns=["key", "value"])

def _flatten_multiindex_columns(cols) -> List[str]:
    """Flatten MultiIndex columns into 'name [unit]' when the second level looks like units."""
    out = []
    for col in cols:
        left = str(col[0]).strip()
        right = str(col[1]).strip()
        if ("/" in right) or ("(" in right) or (")" in right):
            out.append(f"{left} [{right}]")
        else:
            out.append((f"{left} {right}").strip())
    return out

def _clean_columns(cols) -> List[str]:
    return [re.sub(r"\s+", " ", str(c)).strip().replace("\ufeff", "") for c in cols]

def _excel_writer(path: str):
    """Pick an Excel writer engine with fallback."""
    try:
        import xlsxwriter  # noqa
        return pd.ExcelWriter(path, engine="xlsxwriter")
    except Exception:
        try:
            import openpyxl  # noqa
            return pd.ExcelWriter(path, engine="openpyxl")
        except Exception:
            raise RuntimeError(
                "No Excel engine found. Install one of:\n"
                "  pip install xlsxwriter\n"
                "  pip install openpyxl"
            )

# ------------------------- main conversion -------------------------

def convert(input_path: str, output_xlsx: str | None = None) -> str:
    if output_xlsx in ("", False, 0):  # treat falsy as None
        output_xlsx = None

    if output_xlsx is None:
        base, _ = os.path.splitext(input_path)
        output_xlsx = base + ".xlsx"

    # Read raw lines (cp1252 handles EC-Lab special chars)
    with open(input_path, "r", encoding="cp1252", errors="replace") as f:
        lines = f.readlines()

    nb_header = _parse_nb_header_lines(lines)
    header_row_abs, has_units = _find_header_and_units(lines, nb_header)

    # Metadata (everything before the detected header row)
    meta_df = _collect_metadata_pairs(lines, header_row_abs)

    # Prepare pandas parameters
    skiprows = list(range(header_row_abs))   # skip everything before the column-name line
    header_param = [0, 1] if has_units else 0

    # Parse table
    try:
        df = pd.read_csv(
            input_path,
            sep="\t",
            skiprows=skiprows,
            header=header_param,
            engine="python",
            encoding="cp1252",
        )
    except Exception:
        df = pd.read_csv(
            input_path,
            delim_whitespace=True,
            skiprows=skiprows,
            header=header_param,
            engine="python",
            encoding="cp1252",
        )

    # Fix columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _flatten_multiindex_columns(df.columns)
    else:
        df.columns = _clean_columns(df.columns)

    # Write Excel (auto-fit widths when engine supports it)
    out_dir = os.path.dirname(output_xlsx) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    with _excel_writer(output_xlsx) as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        meta_df.to_excel(writer, sheet_name="metadata", index=False)

        # Auto-fit (only works with xlsxwriter)
        try:
            for sheet, frame in [("data", df), ("metadata", meta_df)]:
                ws = writer.sheets[sheet]
                for i, col in enumerate(frame.columns):
                    max_len = max([len(str(col))] + [len(str(v)) for v in frame[col].astype(str).values[:1000]])
                    ws.set_column(i, i, min(max(10, max_len + 2), 60))
        except Exception:
            pass  # openpyxl engine has no set_column

    return output_xlsx

# ------------------------- CLI / IDE entry -------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_eclab_ascii_to_xlsx.py <input.txt|.mpt> [output.xlsx]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_xlsx = sys.argv[2] if len(sys.argv) >= 3 else None
    out = convert(input_path, output_xlsx)
    print(out)

if __name__ == "__main__":
    # For IDE use, you can hardcode paths here, then press Run:
    input_path = r"1.5UC-SRO-BTO-SRO-NSTO(3rd_sample_new)-1MKOH_GraphiteCE_SCE_REF_CV_upvsdown_09_CV_C01.mpt"   # or .mpt (ASCII export)
    output_xlsx = None
    print(convert(input_path, output_xlsx))
    main()
