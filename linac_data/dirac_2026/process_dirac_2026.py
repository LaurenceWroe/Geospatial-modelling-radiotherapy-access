#!/usr/bin/env python
"""Parse the IAEA DIRAC 2026 export, diff it against the July 2025 export used for
the RadMaps paper, carry coordinates over from the audited 2025 file and write a
summary.  Works only on CSV/XLSX files -- no network, no population grids.

Inputs (relative to the repo root):
    linac_data/dirac_2026/DIRAC_2026_raw.xlsx   raw export (no Country column; country header rows)
    linac_data/Database_DIRAC.csv               July 2025 export (original)
    linac_data/Database_DIRAC_fixed.csv         same rows, all coordinates filled after audit

Outputs (linac_data/dirac_2026/):
    Database_DIRAC_2026.csv       tidy 2026 export, 2025 column layout, coords carried over where matched
    geocode_todo_2026.csv         rows still lacking coordinates
    match_log_2026.csv            one row per 2026 centre: how it was matched (exact/fuzzy/none) and MVM delta
    removed_2025.csv              2025 centres with no 2026 counterpart
    near_miss_2026.csv            diagnostic: best unmatched 2025 candidate for each unmatched 2026 row (not matched)
    country_name_map_2026.csv     DIRAC 2026 country label -> harmonised (2025) label
    header_check_2026.csv         per-country header Count/Total/Latest vs parsed values
    SUMMARY.md                    all numbers + data-quality notes

Run:  /opt/anaconda3/bin/python linac_data/dirac_2026/process_dirac_2026.py
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW_XLSX = HERE / "DIRAC_2026_raw.xlsx"
OLD_CSV = ROOT / "linac_data" / "Database_DIRAC.csv"
FIXED_CSV = ROOT / "linac_data" / "Database_DIRAC_fixed.csv"

COLS = [
    "Country", "City", "Operator Name",
    "He Photon And Electron Beam Rt", "Proton Ion Therapy", "XRay Generator",
    "Brachy Therapy Inc El", "Last Update", "Latitude", "Longitude",
]
MVM = "He Photon And Electron Beam Rt"
COUNT_COLS = ["He Photon And Electron Beam Rt", "Proton Ion Therapy", "XRay Generator", "Brachy Therapy Inc El"]
FUZZY_THRESHOLD = 0.9
OP_ONLY_THRESHOLD = 0.95   # operator-name-only similarity that overrides a city disagreement ...
CITY_MIN_RATIO = 0.85      # ... provided the cities are at least this similar (Dagupan Cty/City yes, Osaka-shi/Saga-shi no)
CITY_MIN_RATIO_SAME_OP = 0.8  # when the operator name is identical a looser city similarity suffices (Mit Ghamr/Meet Ghamr)
NEAR_MISS_THRESHOLD = 0.75  # diagnostic only: best unmatched candidate reported, never matched

# DIRAC 2026 country label -> label used in the 2025 export (Database_DIRAC.csv),
# so the output is a drop-in replacement for the paper's file and so that the
# diff compares like with like.  Labels not listed here are identical in both.
COUNTRY_MAP_2026_TO_2025 = {
    "Aruba (Netherlands)": "Aruba",
    "People's Republic of Bangladesh": "Bangladesh",
    "Brunei Darussalam": "Brunei",
    "People's Republic of China": "China",
    "Cote D'Ivoire": "Cote D’Ivoire",
    "Democratic Republic of the Congo": "Democratic Republic of Congo",
    "Guadeloupe (France)": "Guadeloupe",
    "Hong Kong SAR (China)": "Hong Kong",
    "Islamic Republic of Iran": "Iran",
    "Republic of Kyrgyzstan": "Kyrgyzstan",
    "Lao People's Democratic Republic": "Laos",
    "Macao SAR (China)": "Macao",
    "Republic of North Macedonia": "Macedonia",
    "Martinique (France)": "Martinique",
    "Republic of Moldova": "Moldova",
    "Democratic People's Republic of Korea": "North Korea",
    "Puerto Rico (US)": "Puerto Rico",
    "Reunion (France)": "Reunion",
    "Russian Federation": "Russia",
    "Slovak Republic": "Slovakia",
    "Republic of Slovenia": "Slovenia",
    "Republic of Korea": "South Korea",
    "Syrian Arab Republic": "Syria",
    "Taiwan (China)": "Taiwan",
    "United States of America": "USA",
}

SIX = ["Australia", "India", "Nigeria", "Oman", "United Kingdom", "USA"]


# --------------------------------------------------------------------------- helpers
def norm(s) -> str:
    """lowercase, strip accents, drop punctuation, collapse whitespace."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fmt_int(v) -> str:
    return "" if pd.isna(v) else str(int(v))


# --------------------------------------------------------------------------- 1. parse 2026
raw = pd.read_excel(RAW_XLSX, dtype=str)
raw_rows = len(raw)
hdr_re = re.compile(r"^Country: (.+?) \(Count: (\d+)\)$")

records, headers = [], []
current = None
for i, r in raw.iterrows():
    city = r["City"]
    m = hdr_re.match(str(city)) if isinstance(city, str) else None
    if m:
        current = {
            "raw_country": m.group(1),
            "country": COUNTRY_MAP_2026_TO_2025.get(m.group(1).strip(), m.group(1).strip()),
            "header_count": int(m.group(2)),
            "header_latest": r["Last Update"],
            "xlsx_row": i + 2,
        }
        for c in COUNT_COLS:
            mt = re.match(r"^Total: (\d+)$", str(r[c]))
            current[f"header_total::{c}"] = int(mt.group(1)) if mt else None
        headers.append(current)
        continue
    if current is None:
        raise RuntimeError(f"data row before any country header at xlsx row {i + 2}")
    records.append({
        "Country": current["country"],
        "DIRAC country label": current["raw_country"],
        "City": city,
        "Operator Name": r["Operator Name"],
        **{c: r[c] for c in COUNT_COLS},
        "Last Update": r["Last Update"],
        "xlsx_row": i + 2,
    })

new = pd.DataFrame(records)
for c in COUNT_COLS + ["Last Update"]:
    nonnum = new[c].notna() & to_num(new[c]).isna()
    if nonnum.any():
        print(f"WARNING non-numeric values in {c}:", new.loc[nonnum, c].tolist())
    new[c] = to_num(new[c])
new["Latitude"] = np.nan
new["Longitude"] = np.nan

# header verification
hdr_rows = []
for h in headers:
    sub = new[new["DIRAC country label"] == h["raw_country"]]
    row = {
        "DIRAC country label": h["raw_country"], "Country": h["country"],
        "header_count": h["header_count"], "parsed_rows": len(sub),
        "count_ok": h["header_count"] == len(sub),
    }
    for c in COUNT_COLS:
        tot = h[f"header_total::{c}"]
        parsed = int(sub[c].sum())
        row[f"header_total::{c}"] = tot
        row[f"parsed_total::{c}"] = parsed
        row[f"total_ok::{c}"] = tot == parsed
    ml = re.match(r"^Latest: (\d+)$", str(h["header_latest"]))
    latest = int(ml.group(1)) if ml else None
    parsed_latest = int(sub["Last Update"].max()) if sub["Last Update"].notna().any() else None
    row["header_latest"] = latest
    row["parsed_latest"] = parsed_latest
    row["latest_ok"] = latest == parsed_latest
    hdr_rows.append(row)
hdr_check = pd.DataFrame(hdr_rows)
hdr_check.to_csv(HERE / "header_check_2026.csv", index=False)
hdr_mismatch = hdr_check[
    ~hdr_check["count_ok"] | ~hdr_check[[f"total_ok::{c}" for c in COUNT_COLS]].all(axis=1) | ~hdr_check["latest_ok"]
]

# --------------------------------------------------------------------------- 2. load 2025
old = pd.read_csv(OLD_CSV, dtype=str)
fixed = pd.read_csv(FIXED_CSV, dtype=str)
assert len(old) == len(fixed)
assert (old[["Country", "City", "Operator Name"]].fillna("") == fixed[["Country", "City", "Operator Name"]].fillna("")).all().all(), \
    "Database_DIRAC.csv and Database_DIRAC_fixed.csv are not row-aligned"
old["old_idx"] = old.index

# Repair the 4 rows in the 2025 CSV where an unquoted comma in the operator name
# shifted every later field one column to the right (operator tail sits in the MVM
# column, counts shifted, Last Update holds the brachy count and reads "0").
shifted_mask = old[MVM].notna() & to_num(old[MVM]).isna()
old_shifted = old[shifted_mask].copy()
old_rep = old.copy()
for i in old_rep[shifted_mask].index:
    r = old_rep.loc[i]
    old_rep.at[i, "Operator Name"] = f"{r['Operator Name']},{r[MVM]}"
    old_rep.at[i, MVM] = r["Proton Ion Therapy"]
    old_rep.at[i, "Proton Ion Therapy"] = r["XRay Generator"]
    old_rep.at[i, "XRay Generator"] = r["Brachy Therapy Inc El"]
    old_rep.at[i, "Brachy Therapy Inc El"] = r["Last Update"]
    old_rep.at[i, "Last Update"] = np.nan
for c in COUNT_COLS + ["Last Update"]:
    old_rep[c] = to_num(old_rep[c])
old_mvm_raw_sum = int(to_num(old[MVM]).sum())      # as the paper pipeline would read it (coerce -> NaN)
old_mvm_rep_sum = int(old_rep[MVM].sum())

# --------------------------------------------------------------------------- 3. match
for df in (new, old_rep):
    df["city_n"] = df["City"].map(norm)
    df["op_n"] = df["Operator Name"].map(norm)
    df["key"] = df["city_n"] + "||" + df["op_n"]

new["match_old_idx"] = -1
new["match_method"] = "none"
new["match_ratio"] = np.nan
used_old = set()

# exact pass (multiset: a duplicated key in one file is matched row-by-row in order)
for country, g_new in new.groupby("Country", sort=False):
    g_old = old_rep[old_rep["Country"] == country]
    pool = defaultdict(list)
    for oi, k in zip(g_old["old_idx"], g_old["key"]):
        pool[k].append(oi)
    for ni, k in zip(g_new.index, g_new["key"]):
        if pool.get(k):
            oi = pool[k].pop(0)
            new.at[ni, "match_old_idx"] = oi
            new.at[ni, "match_method"] = "exact"
            new.at[ni, "match_ratio"] = 1.0
            used_old.add(oi)

# fuzzy pass on the remainder, within country, greedy by descending ratio
for country, g_new in new[new["match_method"] == "none"].groupby("Country", sort=False):
    g_old = old_rep[(old_rep["Country"] == country) & ~old_rep["old_idx"].isin(used_old)]
    if g_old.empty:
        continue
    cands = []
    for ni, kn, cn, on in zip(g_new.index, g_new["key"], g_new["city_n"], g_new["op_n"]):
        for oi, ko, co, oo in zip(g_old["old_idx"], g_old["key"], g_old["city_n"], g_old["op_n"]):
            ratio = difflib.SequenceMatcher(None, kn, ko).ratio()
            if ratio < FUZZY_THRESHOLD:
                continue
            # guard against near-identical names in *different* cities (e.g. Matsue vs Matsudo
            # City Hospital): the city must agree, or one contain the other, or the operator
            # name must be near-identical on its own.
            city_ok = cn == co or (cn and co and (cn in co or co in cn))
            city_ratio = difflib.SequenceMatcher(None, cn, co).ratio()
            op_ratio = difflib.SequenceMatcher(None, on, oo).ratio()
            op_ok = ((on == oo and city_ratio >= CITY_MIN_RATIO_SAME_OP)
                     or (op_ratio >= OP_ONLY_THRESHOLD and city_ratio >= CITY_MIN_RATIO))
            if city_ok or op_ok:
                cands.append((ratio, ni, oi))
    cands.sort(reverse=True)
    taken_new = set()
    for ratio, ni, oi in cands:
        if ni in taken_new or oi in used_old:
            continue
        new.at[ni, "match_old_idx"] = oi
        new.at[ni, "match_method"] = "fuzzy"
        new.at[ni, "match_ratio"] = ratio
        used_old.add(oi)
        taken_new.add(ni)

matched = new["match_old_idx"] >= 0
mi = new.loc[matched, "match_old_idx"].astype(int).values
new.loc[matched, "old_City"] = old_rep.loc[mi, "City"].values
new.loc[matched, "old_Operator Name"] = old_rep.loc[mi, "Operator Name"].values
new.loc[matched, "old_MVM"] = old_rep.loc[mi, MVM].values
new.loc[matched, "old_Last Update"] = old_rep.loc[mi, "Last Update"].values
new["mvm_delta"] = new[MVM] - new["old_MVM"]

# --------------------------------------------------------------------------- 4. coordinates
fixed_lat = to_num(fixed["Latitude"])
fixed_lon = to_num(fixed["Longitude"])
new.loc[matched, "Latitude"] = fixed_lat.loc[mi].values
new.loc[matched, "Longitude"] = fixed_lon.loc[mi].values
has_coord = new["Latitude"].notna() & new["Longitude"].notna()

out = new[COLS].copy()
for c in COUNT_COLS + ["Last Update"]:
    out[c] = out[c].map(fmt_int)
out.to_csv(HERE / "Database_DIRAC_2026.csv", index=False)

todo = new.loc[~has_coord, ["Country", "City", "Operator Name"]]
todo.to_csv(HERE / "geocode_todo_2026.csv", index=False)

removed = old_rep[~old_rep["old_idx"].isin(used_old)]

# diagnostic: for every unmatched 2026 row, the best unmatched 2025 candidate in the same
# country (ratio >= NEAR_MISS_THRESHOLD, or identical operator under a different city).
near_rows = []
for country, g_new in new[new["match_method"] == "none"].groupby("Country", sort=False):
    g_old = removed[removed["Country"] == country]
    if g_old.empty:
        continue
    for ni, kn, cn, on in zip(g_new.index, g_new["key"], g_new["city_n"], g_new["op_n"]):
        best = None
        for oi, ko, co, oo in zip(g_old["old_idx"], g_old["key"], g_old["city_n"], g_old["op_n"]):
            ratio = difflib.SequenceMatcher(None, kn, ko).ratio()
            same_op = on == oo and on != ""
            if ratio >= NEAR_MISS_THRESHOLD or same_op:
                score = 1.0 if same_op else ratio
                if best is None or score > best[0]:
                    best = (score, oi, same_op, ratio)
        if best:
            o = old_rep.loc[best[1]]
            near_rows.append({
                "Country": country, "City": new.at[ni, "City"], "Operator Name": new.at[ni, "Operator Name"],
                MVM: new.at[ni, MVM], "old_City": o["City"], "old_Operator Name": o["Operator Name"],
                "old_MVM": o[MVM], "combined_ratio": round(best[3], 3), "same_operator_diff_city": best[2],
                "old_Latitude": fixed_lat.loc[best[1]], "old_Longitude": fixed_lon.loc[best[1]],
            })
near_miss = pd.DataFrame(near_rows)
near_miss.to_csv(HERE / "near_miss_2026.csv", index=False)
removed[["Country", "City", "Operator Name"] + COUNT_COLS + ["Last Update"]].to_csv(HERE / "removed_2025.csv", index=False)

log_cols = ["xlsx_row", "Country", "DIRAC country label", "City", "Operator Name", MVM, "Last Update",
            "match_method", "match_ratio", "match_old_idx", "old_City", "old_Operator Name", "old_MVM",
            "old_Last Update", "mvm_delta", "Latitude", "Longitude"]
new[log_cols].to_csv(HERE / "match_log_2026.csv", index=False)

pd.DataFrame({
    "DIRAC 2026 label": [h["raw_country"] for h in headers],
    "harmonised (2025) label": [h["country"] for h in headers],
    "renamed": [h["raw_country"].strip() in COUNTRY_MAP_2026_TO_2025 for h in headers],
}).to_csv(HERE / "country_name_map_2026.csv", index=False)

# --------------------------------------------------------------------------- 5. statistics
old_countries = set(old_rep["Country"])
new_countries = set(new["Country"])
countries_added = sorted(new_countries - old_countries)
countries_removed = sorted(old_countries - new_countries)

n_exact = int((new["match_method"] == "exact").sum())
n_fuzzy = int((new["match_method"] == "fuzzy").sum())
n_new = int((new["match_method"] == "none").sum())
n_removed = len(removed)

changed = new[matched & (new["mvm_delta"] != 0)]
n_changed = len(changed)
plus = int(changed.loc[changed["mvm_delta"] > 0, "mvm_delta"].sum())
minus = int(changed.loc[changed["mvm_delta"] < 0, "mvm_delta"].sum())
n_up = int((changed["mvm_delta"] > 0).sum())
n_down = int((changed["mvm_delta"] < 0).sum())
mvm_new_centres = int(new.loc[~matched, MVM].sum())
mvm_removed = int(removed[MVM].sum())
mvm_new_total = int(new[MVM].sum())

years = new["Last Update"].value_counts(dropna=False).sort_index()


def country_table(names):
    rows = []
    for c in names:
        gn = new[new["Country"] == c]
        go = old_rep[old_rep["Country"] == c]
        gr = removed[removed["Country"] == c]
        gc = gn[(gn["match_old_idx"] >= 0) & (gn["mvm_delta"] != 0)]
        rows.append({
            "Country": c,
            "rows 2025": len(go), "rows 2026": len(gn),
            "MVM 2025": int(go[MVM].sum()), "MVM 2026": int(gn[MVM].sum()),
            "matched exact": int((gn["match_method"] == "exact").sum()),
            "matched fuzzy": int((gn["match_method"] == "fuzzy").sum()),
            "new": int((gn["match_method"] == "none").sum()),
            "removed": len(gr),
            "changed": len(gc),
            "MVM +": int(gc.loc[gc["mvm_delta"] > 0, "mvm_delta"].sum()),
            "MVM -": int(gc.loc[gc["mvm_delta"] < 0, "mvm_delta"].sum()),
            "coords carried": int((gn["Latitude"].notna()).sum()),
            "need geocoding": int((gn["Latitude"].isna()).sum()),
        })
    return pd.DataFrame(rows)


six = country_table(SIX)
six.to_csv(HERE / "six_country_table_2026.csv", index=False)
all_ct = country_table(sorted(new_countries | old_countries))
all_ct.to_csv(HERE / "per_country_diff_2026.csv", index=False)

todo_by_country = todo["Country"].value_counts()

# data-quality oddities
dup_exact = new[new.duplicated(["Country", "City", "Operator Name"] + COUNT_COLS + ["Last Update"], keep=False)]
dup_key = new[new.duplicated(["Country", "City", "Operator Name"], keep=False)]
blank_city = new[new["City"].isna() | (new["City"].astype(str).str.strip() == "")]
blank_op = new[new["Operator Name"].isna() | (new["Operator Name"].astype(str).str.strip() == "")]
blank_lu = new[new["Last Update"].isna()]
lu_zero = new[new["Last Update"] == 0]
lu_old = new[new["Last Update"] < 2005]
all_zero = new[(new[COUNT_COLS].fillna(0).sum(axis=1) == 0)]
ws = new[(new["City"].astype(str) != new["City"].astype(str).str.strip()) |
         (new["Operator Name"].astype(str) != new["Operator Name"].astype(str).str.strip())]
raw_labels_ws = [h["raw_country"] for h in headers if h["raw_country"] != h["raw_country"].strip()]
old_dup_key = old_rep[old_rep.duplicated(["Country", "City", "Operator Name"], keep=False)]
old_lu_zero = old[old["Last Update"] == "0"]
old_lu_blank = old[old["Last Update"].isna()]
new_in_matched_dup = new[matched & new.duplicated(["Country", "City", "Operator Name"], keep=False)]
fuzzy_rows = new[new["match_method"] == "fuzzy"]
fuzzy_mvm_changed = fuzzy_rows[fuzzy_rows["mvm_delta"] != 0]
dup_flag = new[new["Operator Name"].astype(str).str.contains("duplicat", case=False)]
same_op_diff_city = near_miss[near_miss["same_operator_diff_city"]] if len(near_miss) else near_miss


# --------------------------------------------------------------------------- 6. SUMMARY.md
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def rows_md(df, cols, limit=None):
    d = df[cols].copy()
    if limit is not None and len(d) > limit:
        return md_table(d.head(limit)) + f"\n\n(first {limit} of {len(d)})"
    return md_table(d) if len(d) else "_none_"


S = []
S.append("# DIRAC 2026 export: parse, diff against July 2025, coordinate carry-over\n")
S.append(f"Generated by `process_dirac_2026.py`. Source `DIRAC_2026_raw.xlsx` ({raw_rows} sheet rows = "
         f"{len(headers)} country header rows + {len(new)} centre rows). Reference: `linac_data/Database_DIRAC.csv` "
         f"({len(old)} rows) / `Database_DIRAC_fixed.csv`.\n")

S.append("## 1. Parsing and header verification\n")
S.append(f"- Country header rows: {len(headers)}; centre rows parsed: {len(new)}.")
S.append(f"- Header `Count:` matches parsed row count for {int(hdr_check['count_ok'].sum())}/{len(hdr_check)} countries.")
for c in COUNT_COLS:
    S.append(f"- Header `Total:` for **{c}** matches parsed column sum for {int(hdr_check[f'total_ok::{c}'].sum())}/{len(hdr_check)} countries.")
S.append(f"- Header `Latest:` matches max(Last Update) for {int(hdr_check['latest_ok'].sum())}/{len(hdr_check)} countries.")
if len(hdr_mismatch):
    S.append("\nMismatches:\n")
    S.append(md_table(hdr_mismatch))
else:
    S.append("- No header mismatches.")
S.append("\nCountry labels: DIRAC 2026 uses longer official labels for 25 countries (e.g. `United States of America`, "
         "`Russian Federation`, `Republic of Korea`). The `Country` column in `Database_DIRAC_2026.csv` uses the 2025 "
         "export's labels (e.g. `USA`, `Russia`, `South Korea`) so the file is a drop-in replacement for "
         "`Database_DIRAC.csv` and so that the diff compares like with like. Full map: `country_name_map_2026.csv`.\n")

S.append("## 2. Headline numbers\n")
S.append(md_table(pd.DataFrame([
    {"": "centre rows", "2025": len(old), "2026": len(new)},
    {"": "countries", "2025": len(old_countries), "2026": len(new_countries)},
    {"": "MVMs (He Photon And Electron Beam Rt, sum)", "2025": f"{old_mvm_rep_sum} ({old_mvm_raw_sum} if the 4 column-shifted 2025 rows are read as NaN)", "2026": mvm_new_total},
    {"": "Proton/ion", "2025": int(old_rep['Proton Ion Therapy'].sum()), "2026": int(new['Proton Ion Therapy'].sum())},
    {"": "X-ray generators", "2025": int(old_rep['XRay Generator'].sum()), "2026": int(new['XRay Generator'].sum())},
    {"": "Brachytherapy", "2025": int(old_rep['Brachy Therapy Inc El'].sum()), "2026": int(new['Brachy Therapy Inc El'].sum())},
])))
S.append("")
S.append(f"- Countries added in 2026: {len(countries_added)} -> {countries_added if countries_added else 'none'}")
S.append(f"- Countries removed in 2026: {len(countries_removed)} -> {countries_removed if countries_removed else 'none'}")
S.append("")
S.append("## 3. Centre matching (within country, normalised city + operator)\n")
S.append(f"- Exact matches: {n_exact}")
S.append(f"- Fuzzy matches (difflib ratio >= {FUZZY_THRESHOLD}): {n_fuzzy}")
S.append(f"- Total matched: {n_exact + n_fuzzy} of {len(new)} 2026 rows ({(n_exact + n_fuzzy) / len(new):.1%})")
S.append(f"- New in 2026 (no 2025 counterpart): {n_new} rows, carrying {mvm_new_centres} MVMs")
S.append(f"- Absent in 2026 (2025 rows with no 2026 counterpart): {n_removed} rows, carrying {mvm_removed} MVMs (see `removed_2025.csv`)")
S.append(f"- Matched centres whose MVM count changed: {n_changed} ({n_up} up, {n_down} down); MVM delta +{plus} / {minus} = net {plus + minus}")
S.append(f"- Net MVM change 2025 -> 2026: {mvm_new_total - old_mvm_rep_sum} "
         f"(= new centres {mvm_new_centres} - removed {mvm_removed} + changed {plus + minus})")
S.append(f"- Fuzzy rule: combined `city||operator` ratio >= {FUZZY_THRESHOLD}, and additionally the normalised city must be equal/contained "
         f"or the operator-only ratio must be >= {OP_ONLY_THRESHOLD} with city ratio >= {CITY_MIN_RATIO} (>= {CITY_MIN_RATIO_SAME_OP} when the operator is identical) (this rejects e.g. *Matsue City Hospital* vs *Matsudo City Hospital* and the Japanese NHO hospitals that share a long prefix).")
S.append(f"- Near misses (diagnostic only, NOT matched, `near_miss_2026.csv`): {len(near_miss)} unmatched 2026 rows have an unmatched 2025 "
         f"row in the same country with combined ratio >= {NEAR_MISS_THRESHOLD} or an identical operator name under a different city "
         f"({len(same_op_diff_city)} of the latter). These are probably renames / city corrections; review before geocoding.")
S.append("")
S.append("Identical operator, different city (2025 -> 2026):\n")
S.append(rows_md(same_op_diff_city, ["Country", "old_City", "City", "Operator Name", "old_MVM", MVM]))
S.append("")
S.append(f"Other near misses (combined ratio {NEAR_MISS_THRESHOLD}-{FUZZY_THRESHOLD}, or >= {FUZZY_THRESHOLD} but rejected by the city guard):\n")
S.append(rows_md(near_miss[~near_miss["same_operator_diff_city"]].sort_values("combined_ratio", ascending=False) if len(near_miss) else near_miss,
                 ["Country", "City", "Operator Name", "old_City", "old_Operator Name", "combined_ratio", "old_MVM", MVM]))
S.append("")
S.append("Fuzzy matches (for review):\n")
S.append(rows_md(fuzzy_rows.assign(match_ratio=fuzzy_rows["match_ratio"].round(3)),
                 ["Country", "City", "Operator Name", "old_City", "old_Operator Name", "match_ratio", "mvm_delta"]))
S.append("")
S.append("## 4. Last Update year distribution (2026 export)\n")
S.append(md_table(pd.DataFrame({"Last Update": [("blank" if pd.isna(k) else int(k)) for k in years.index],
                                "rows": years.values})))
S.append("")
S.append("## 5. Coordinates\n")
S.append(f"- Coordinates carried over from `Database_DIRAC_fixed.csv`: {int(has_coord.sum())} rows")
S.append(f"- Rows still lacking coordinates (`geocode_todo_2026.csv`): {int((~has_coord).sum())} rows in {len(todo_by_country)} countries")
S.append("")
S.append(md_table(todo_by_country.rename_axis("Country").reset_index(name="rows needing geocoding")))
S.append("")
S.append("## 6. Six-country table\n")
S.append(md_table(six))
S.append("\nAll countries: `per_country_diff_2026.csv`.\n")

S.append("## 7. Data-quality oddities\n")
S.append("### 2026 export")
S.append(f"- Exact duplicate centre rows (all fields identical): {len(dup_exact)} rows in {dup_exact.duplicated(['Country','City','Operator Name'] + COUNT_COLS + ['Last Update']).sum()} pairs")
S.append(rows_md(dup_exact, ["Country", "City", "Operator Name", MVM, "Last Update"]))
S.append(f"\n- Rows sharing (Country, City, Operator Name) but differing elsewhere or identical: {len(dup_key)} rows "
         f"({len(dup_key) - len(dup_exact)} beyond the exact duplicates)")
S.append(rows_md(dup_key[~dup_key.index.isin(dup_exact.index)], ["Country", "City", "Operator Name", MVM, "Proton Ion Therapy", "XRay Generator", "Brachy Therapy Inc El", "Last Update"]))
S.append(f"\n- Blank City: {len(blank_city)} rows")
S.append(rows_md(blank_city, ["Country", "Operator Name", MVM, "Last Update"]))
S.append(f"\n- Blank Operator Name: {len(blank_op)} rows")
S.append(rows_md(blank_op, ["Country", "City", MVM, "Last Update"]))
S.append(f"\n- Blank Last Update: {len(blank_lu)} rows")
S.append(rows_md(blank_lu, ["Country", "City", "Operator Name", MVM]))
S.append(f"\n- Last Update == 0: {len(lu_zero)} rows")
S.append(f"- Last Update < 2005: {len(lu_old)} rows")
S.append(rows_md(lu_old.sort_values("Last Update"), ["Country", "City", "Operator Name", MVM, "Last Update"]))
S.append(f"\n- Non-numeric machine counts: none (all four count columns parse as integers in every centre row).")
S.append(f"- Rows with all four machine counts == 0: {len(all_zero)}")
S.append(rows_md(all_zero, ["Country", "City", "Operator Name", "Last Update"], limit=40))
S.append(f"\n- Leading/trailing whitespace in City or Operator Name: {len(ws)} rows (stripped only in the normalised key; raw text kept in the CSV)")
S.append(f"- Country header labels with trailing whitespace: {raw_labels_ws if raw_labels_ws else 'none'}")
S.append(f"- Matched 2026 rows that are themselves (Country, City, Operator) duplicates (matched one-to-one in order): {len(new_in_matched_dup)}")
S.append(f"- Fuzzy-matched rows whose MVM count also changed (possible false match, check): {len(fuzzy_mvm_changed)}")
S.append(f"- Operator names flagged as duplicates by DIRAC itself (contain 'DUPLICATED'): {len(dup_flag)} rows -- consider dropping")
S.append(rows_md(dup_flag, ["Country", "City", "Operator Name", MVM, "Last Update", "match_method"]))
S.append("\n### 2025 export (reference)")
S.append("- Four USA centres sat under the wrong city in 2025 and DIRAC has corrected them in 2026 (UNC Lenoir Memorial Hospital "
         "'Chicago' -> Kinston; Inova Loudoun Hospital 'Fresno' -> Leesburg; Novant Health Forsyth Medical Center 'Columbia' -> Winston-Salem; "
         "VA Medical Center-Philadelphia 'Winston Salem' -> Philadelphia). In `Database_DIRAC_fixed.csv` the first three carry the coordinates of "
         "the wrong city (Chicago, Fresno, Columbia SC), i.e. 8 MVMs were mis-located in the paper's input; the fourth already had Philadelphia "
         "coordinates. They appear as new+removed here and need fresh geocoding.")

S.append(f"- 4 rows in `Database_DIRAC.csv` have an unquoted comma in the operator name that shifts every later field one column right "
         f"(operator tail in the MVM column, counts shifted, `Last Update` holds the brachy count and reads `0`). "
         f"These are the 4 `Last Update == 0` rows. They were repaired in memory for the diff; their true Last Update is lost. "
         f"Reading the file with `pd.to_numeric(errors='coerce')` as the pipeline does silently drops "
         f"{int(old_rep.loc[shifted_mask, MVM].sum())} MVMs from these rows:")
S.append(rows_md(old_shifted, ["Country", "City", "Operator Name", MVM, "Proton Ion Therapy", "XRay Generator", "Brachy Therapy Inc El", "Last Update"]))
S.append(f"\n- Blank Last Update in 2025: {len(old_lu_blank)} rows (same centre is blank in 2026).")
S.append(f"- Rows sharing (Country, City, Operator Name) in 2025: {len(old_dup_key)}")
S.append(f"- `Database_DIRAC_fixed.csv` is row-aligned with `Database_DIRAC.csv` (checked) and has no blank coordinates.")
S.append("\n## 8. Output files\n")
for f in ["DIRAC_2026_raw.xlsx", "Database_DIRAC_2026.csv", "geocode_todo_2026.csv", "match_log_2026.csv", "removed_2025.csv",
          "near_miss_2026.csv", "country_name_map_2026.csv", "header_check_2026.csv", "six_country_table_2026.csv", "per_country_diff_2026.csv",
          "process_dirac_2026.py", "SUMMARY.md"]:
    S.append(f"- `linac_data/dirac_2026/{f}`")
(HERE / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8")

# --------------------------------------------------------------------------- console
print(f"2025: {len(old)} rows, {len(old_countries)} countries, MVM {old_mvm_rep_sum} (raw-coerced {old_mvm_raw_sum})")
print(f"2026: {len(new)} rows, {len(new_countries)} countries, MVM {mvm_new_total}")
print(f"countries added {countries_added} removed {countries_removed}")
print(f"matched exact {n_exact} fuzzy {n_fuzzy} new {n_new} removed {n_removed}")
print(f"changed {n_changed} (+{plus} / {minus}); new-centre MVM {mvm_new_centres}; removed MVM {mvm_removed}")
print(f"coords carried {int(has_coord.sum())}; need geocoding {int((~has_coord).sum())} in {len(todo_by_country)} countries")
print(f"header mismatches: {len(hdr_mismatch)}")
print(six.to_string(index=False))
