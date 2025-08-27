# ---- INITIAL CHECKS ---- #
import importlib.util
import subprocess
import sys

# Packages and their import names (sometimes different from pip names)
packages = {
    "selenium": "selenium",
    "webdriver-manager": "webdriver_manager",
    "requests": "requests",
    "pandas": "pandas",
    "tabula-py": "tabula",
    "pdfplumber": "pdfplumber",
    "pyarrow": "pyarrow",
}

for pip_name, import_name in packages.items():
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
    else:
        print(f"{pip_name} already installed.")

# ---- Actual Script ---- #

import os, re, time, pathlib, concurrent.futures, tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests
import pandas as pd
import numpy as np


# --- selenium just to discover the PDF links reliably (page is JS-rendered) ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# --- table extraction ---
import tabula      # Java-backed; very good on these PDFs
import pdfplumber  # fallback if a table fails in tabula

FACTSHEETS_INDEX = "https://gco.iarc.who.int/today/en/fact-sheets-populations#countries"
OUT_DIR = pathlib.Path("globocan_factsheets")
PDF_DIR = OUT_DIR / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)

def list_population_factsheet_pdfs(timeout=30):
    """Return list of (country_text, href) for all population fact-sheet PDFs."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    try:
        driver.get(FACTSHEETS_INDEX)

        # Some sessions show a cookie banner; try to accept if present.
        try:
            WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept') or contains(., 'accept')]"))
            ).click()
        except Exception:
            pass

        # Wait for links to render
        WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.TAG_NAME, "a"))
        )
        links = driver.find_elements(By.CSS_SELECTOR, "a[href$='.pdf']")
        pdfs = []
        for a in links:
            href = a.getAttribute("href") if hasattr(a, "getAttribute") else a.get_attribute("href")
            text = a.text.strip()
            if href and "/media/globocan/factsheets/populations/" in href and href.endswith(".pdf"):
                pdfs.append((text or pathlib.Path(urlparse(href).path).name, href))
        # Deduplicate
        seen = set(); out = []
        for t,h in pdfs:
            if h not in seen:
                seen.add(h); out.append((t,h))
        return out
    finally:
        driver.quit()

def safe_filename(url):
    return pathlib.Path(urlparse(url).path).name

def download_one(url, dest_dir=PDF_DIR, sleep_sec=0.2):
    fn = dest_dir / safe_filename(url)
    if fn.exists() and fn.stat().st_size > 0:
        return fn
    hdrs = {"User-Agent": "research/archival (contact: your-email@example.com)"}
    with requests.get(url, headers=hdrs, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(fn, "wb") as f:
            for chunk in r.iter_content(1<<15):
                if chunk: f.write(chunk)
    time.sleep(sleep_sec)    # be polite
    return fn

# ---------- Helpers ----------

def filename_to_meta(path):
    """
    Derive country name and numeric code from filename like:
    '840-united-states-of-america-fact-sheet.pdf'
    """
    name = pathlib.Path(path).name
    m = re.match(r"(\d+)-([a-z0-9\-]+)-fact-sheet\.pdf", name)
    if not m:
        return {"pop_code": None, "country": name.replace("-fact-sheet.pdf","")}
    pop_code = int(m.group(1))
    country = m.group(2).replace("-", " ").title()
    # Handle some known acronyms better
    country = country.replace(" Usa", " USA").replace(" Uk", " UK")
    return {"pop_code": pop_code, "country": country}


def _clean_columns(cols):
    return [re.sub(r"\s+", " ", str(c)).strip() if c is not None else "" for c in cols]

def _looks_like_two_row_header(df: pd.DataFrame) -> bool:
    if df.shape[0] < 2:
        return False
    row0 = " ".join(_clean_columns(df.iloc[0].tolist())).lower()
    row1 = " ".join(_clean_columns(df.iloc[1].tolist())).lower()
    triggers = ["new cases", "deaths", "5 year", "5-year", "prevalence", "cum.risk", "(%)"]
    return any(t in row0 for t in triggers) and any(t in row1 for t in triggers + ["cancer"])

def _combine_two_row_header(df: pd.DataFrame) -> pd.DataFrame:
    major = pd.Series(_clean_columns(df.iloc[0].tolist()))
    minor = pd.Series(_clean_columns(df.iloc[1].tolist()))
    # Forward-fill major headers and force the first to "Cancer"
    major = major.replace({"": np.nan}).ffill().fillna("")
    if not major.iloc[0] or "cancer" in minor.iloc[0].lower():
        major.iloc[0] = "Cancer"
    # Build combined header
    combined = []
    for M, m in zip(major, minor):
        if not M: 
            combined.append(m or "")
        elif not m or m.lower() == M.lower():
            combined.append(M)
        else:
            combined.append(f"{M} {m}")
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = _clean_columns(combined)
    return df

def _pick_cancer_column(df: pd.DataFrame) -> str | None:
    # Heuristic: "Cancer" column is mostly text (few numeric-only cells), often includes items like "Breast", "Prostate", "All cancers"
    best_col, best_texty = None, -1
    for c in df.columns:
        series = df[c].astype(str)
        # ratio of cells that contain letters (A–Z)
        texty = (series.str.contains(r"[A-Za-z]", na=False)).mean()
        if texty > best_texty:
            best_texty, best_col = texty, c
    return best_col

def _rename_metrics(columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        low = c.lower()

        # Normalize some tokens
        low = low.replace("5 year", "5-year").replace("per 100 000", "per 100000")
        low = re.sub(r"\s+", " ", low)

        # Map
        if low == "cancer" or "cancer" == low.strip():
            out.append("Cancer"); continue

        def has(*tokens): return all(t in low for t in tokens)
        def any_of(*tokens): return any(t in low for t in tokens)

        if has("new cases") and any_of("number", "no."):
            out.append("Incidence_Number")
        elif has("new cases") and "rank" in low:
            out.append("Incidence_Rank")
        elif has("new cases") and ("(%)" in low or "percent" in low or "percentage" in low or "percent." in low):
            out.append("Incidence_Percent")
        elif has("new cases") and any_of("cum.risk", "cum risk", "cum-risk", "cumulative risk"):
            out.append("Incidence_CumRisk")

        elif "deaths" in low and any_of("number", "no."):
            out.append("Mortality_Number")
        elif "deaths" in low and "rank" in low:
            out.append("Mortality_Rank")
        elif "deaths" in low and ("(%)" in low or "percent" in low or "percentage" in low):
            out.append("Mortality_Percent")
        elif "deaths" in low and any_of("cum.risk", "cum risk", "cum-risk", "cumulative risk"):
            out.append("Mortality_CumRisk")

        elif any_of("5-year prevalence", "5-year-prevalence", "5 year prevalence", "5-year prevalence", "prevalence"):
            if any_of("prop", "per 100000"):
                out.append("Prevalence_per100k")
            elif any_of("number", "no."):
                out.append("Prevalence_Number")
            else:
                out.append(c)  # leave as-is

        else:
            out.append(c)
    return _clean_columns(out)

def _final_numeric_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    if "Cancer" in df.columns:
        df["Cancer"] = df["Cancer"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    num_cols = [c for c in df.columns if c != "Cancer"]
    for c in num_cols:
        df[c] = (
            df[c].astype(str)
                  .str.replace(r"[^\d\.\-]", "", regex=True)  # keep digits, dot, minus
                  .replace({"": np.nan, "-": np.nan, "–": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _standardize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Some Tabula returns include header rows as data; detect and combine
    # Also flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _clean_columns([" ".join([str(x) for x in tpl if str(x) != "None"]) for tpl in df.columns])
    else:
        df.columns = _clean_columns(df.columns)

    # If first rows look like header rows, rebuild header
    if _looks_like_two_row_header(df):
        df = _combine_two_row_header(df)

    # If we still don't have a "Cancer" column, try to promote first row or pick a text-heavy column
    if "Cancer" not in [c.strip().title() for c in df.columns]:
        # Try: if row 0 contains "Cancer" under some blank header, promote row 0 to header
        if df.shape[0] > 0 and any("cancer" in str(x).lower() for x in df.iloc[0].tolist()):
            df.columns = _clean_columns(df.iloc[0].tolist())
            df = df.iloc[1:].reset_index(drop=True)
        # Rename synonyms
        rename_syn = {c: "Cancer" for c in df.columns if re.search(r"^cancer(\s*site)?$", c.strip(), flags=re.I)}
        df = df.rename(columns=rename_syn)
        # If still missing, pick the text-heavy column and call it "Cancer"
        if "Cancer" not in df.columns:
            cand = _pick_cancer_column(df)
            if cand:
                df = df.rename(columns={cand: "Cancer"})

    # Rename metric columns via robust regex rules
    df.columns = _rename_metrics(list(df.columns))

    # Keep and order only the expected columns if present
    wanted = [
        "Cancer",
        "Incidence_Number","Incidence_Rank","Incidence_Percent","Incidence_CumRisk",
        "Mortality_Number","Mortality_Rank","Mortality_Percent","Mortality_CumRisk",
        "Prevalence_Number","Prevalence_per100k"
    ]
    present = [c for c in wanted if c in df.columns]
    if "Cancer" not in present:
        raise ValueError("Could not identify the 'Cancer' column after header reconstruction.")
    df = df[present].copy()

    # Drop rows where Cancer is blank/NaN
    df = df[df["Cancer"].astype(str).str.strip().ne("")].reset_index(drop=True)

    # Numeric cleanup
    df = _final_numeric_cleanup(df)

    # Optional: drop obvious footer/noise rows if they slipped in
    # (You can keep 'All cancers' rows if you want them.)
    # df = df[~df["Cancer"].str.contains(r"^Incidence, Mortality", na=False)]

    return df.reset_index(drop=True)

# ---------- Extraction core ----------

def parse_fact_sheet_table(pdf_path):
    """
    Parse the 'Incidence, Mortality and Prevalence by cancer site' table on page 2.
    Returns a tidy DataFrame or raises on failure.
    """
    # ----- 1) Tabula lattice (with guess) -----
    try:
        dfs = tabula.read_pdf(
            str(pdf_path),
            pages="2",
            lattice=True,
            stream=False,
            guess=True,                 # changed: let Tabula guess cell boundaries
            multiple_tables=True,
            pandas_options={"dtype": str}
        ) or []
        # don't over-filter; keep small candidates too
        dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and d.shape[1] >= 4]
        dfs.sort(key=lambda d: (d.shape[0] * d.shape[1]), reverse=True)
        for cand in dfs:
            try:
                out = _standardize_table(cand)
                if len(out) >= 10 and any(c != "Cancer" for c in out.columns):
                    return out
            except Exception:
                pass
    except Exception:
        pass

    # ----- 2) Tabula stream (whitespace-delimited tables) -----
    try:
        dfs = tabula.read_pdf(
            str(pdf_path),
            pages="2",
            lattice=False,
            stream=True,
            guess=True,
            multiple_tables=True,
            pandas_options={"dtype": str}
        ) or []
        dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and d.shape[1] >= 4]
        dfs.sort(key=lambda d: (d.shape[0] * d.shape[1]), reverse=True)
        for cand in dfs:
            try:
                out = _standardize_table(cand)
                if len(out) >= 10 and any(c != "Cancer" for c in out.columns):
                    return out
            except Exception:
                pass
    except Exception:
        pass

    # 2) Fallback: pdfplumber with lines strategy (robust to some PDFs where Tabula splits merged cells)
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[1]  # page 2 (0-indexed)
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 2,
            "edge_min_length": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "intersection_tolerance": 3,
            # use the *text_* options supported by new pdfplumber:
            "text_x_tolerance": 2,
            "text_y_tolerance": 2,
            "text_keep_blank_chars": True,
            }
        tables = page.extract_tables(table_settings=table_settings) or []
        # choose biggest table
        tables.sort(key=lambda t: (len(t) * len(t[0]) if t and t[0] else 0), reverse=True)
        for t in tables:
            if not t or not t[0]:
                continue
            df = pd.DataFrame(t[1:], columns=_clean_columns(t[0]))
            try:
                out = _standardize_table(df)
                if len(out) >= 10 and any(c != "Cancer" for c in out.columns):
                    return out
            except Exception:
                continue

    raise RuntimeError("Failed to parse the page-2 site-by-metrics table from this PDF.")


def main():
    """print("Discovering fact-sheet PDFs…")
    links = list_population_factsheet_pdfs()
    print(f"Found {len(links)} PDFs")

    # Download politely, a few at a time
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(download_one, href) for _, href in links]
        pdf_paths = [f.result() for f in concurrent.futures.as_completed(futures)]
    pdf_paths.sort()"""
    PDF_DIR = Path("globocan_factsheets/pdf")
    OUT_DIR = Path("globocan_factsheets")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {PDF_DIR.resolve()}")

    print("Parsing page-2 tables… (this may take a few minutes)")
    results = []
    failures = []

    for p in pdf_paths:
        meta = filename_to_meta(p)
        try:
            df = parse_fact_sheet_table(p)
            # annotate with metadata
            df.insert(0, "pop_code", meta["pop_code"])
            df.insert(1, "country", meta["country"])
            df.insert(2, "source_pdf", p.name)
            results.append(df)
            print(f"OK: {p.name} → {len(df)} rows")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failures.append({"pdf": p.name, "error": err})
            print(f"FAILED: {p.name} -> {err}")

    if not results:
        raise SystemExit("No tables parsed. Check Java/Tabula (Java) and pdfplumber and try again.")

    big = pd.concat(results, ignore_index=True, sort=False)

    # Normalize column order: metadata first, then the expected metrics if present
    wanted = [
        "Cancer",
        "Incidence_Number", "Incidence_Rank", "Incidence_Percent", "Incidence_CumRisk",
        "Mortality_Number", "Mortality_Rank", "Mortality_Percent", "Mortality_CumRisk",
        "Prevalence_Number", "Prevalence_per100k",
    ]
    ordered = ["pop_code", "country", "source_pdf"] + [c for c in wanted if c in big.columns]
    # keep any extra columns (if present) at the end
    ordered += [c for c in big.columns if c not in ordered]
    big = big[ordered]

    # Save outputs
    csv_path = OUT_DIR / "globocan2022_fact_sheets_page2.csv"
    pq_path  = OUT_DIR / "globocan2022_fact_sheets_page2.parquet"
    big.to_csv(csv_path, index=False)
    big.to_parquet(pq_path, index=False)

    # Save a failure log if any
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "parse_failures.csv", index=False)

    print(f"Done → {csv_path} "
          f"({len(big)} rows from {len(results)} PDFs; {len(failures)} failures)")
    

if __name__ == "__main__":
    main()
