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

def _clean_columns(cols):
    # Flatten multi-row headers Tabula may produce
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]

def _standardize_table(df):
    df = df.copy()
    df.columns = _clean_columns(df.columns)
    # Heuristic: keep only the big site-by-metrics table (has 'Cancer' in first col)
    if 'Cancer' not in df.columns[0]:
        # Sometimes the first row is headers; promote row 0 to header if it contains 'Cancer'
        if 'Cancer' in str(df.iloc[0, 0]):
            df.columns = _clean_columns(df.iloc[0].tolist())
            df = df.iloc[1:].reset_index(drop=True)
    # Rename common messy headers
    rename_map = {}
    for c in list(df.columns):
        low = c.lower()
        if low.startswith("cancer"):
            rename_map[c] = "Cancer"
        elif "new cases" in low and "cum" in low:
            rename_map[c] = "Incidence_CumRisk"
        elif "new cases" in low and "rank" in low:
            rename_map[c] = "Incidence_Rank"
        elif "new cases" in low and ("percent" in low or "(%)" in low):
            rename_map[c] = "Incidence_Percent"
        elif "new cases" in low or "incidence" in low:
            rename_map[c] = "Incidence_Number"
        elif "deaths" in low and "cum" in low:
            rename_map[c] = "Mortality_CumRisk"
        elif "deaths" in low and "rank" in low:
            rename_map[c] = "Mortality_Rank"
        elif "deaths" in low and ("percent" in low or "(%)" in low):
            rename_map[c] = "Mortality_Percent"
        elif "deaths" in low:
            rename_map[c] = "Mortality_Number"
        elif "prevalence" in low and ("per 100" in low or "prop" in low):
            rename_map[c] = "Prevalence_per100k"
        elif "prevalence" in low and "number" in low:
            rename_map[c] = "Prevalence_Number"
    df = df.rename(columns=rename_map)
    # Keep only expected columns if present
    wanted = ["Cancer","Incidence_Number","Incidence_Rank","Incidence_Percent","Incidence_CumRisk",
              "Mortality_Number","Mortality_Rank","Mortality_Percent","Mortality_CumRisk",
              "Prevalence_Number","Prevalence_per100k"]
    keep = [c for c in wanted if c in df.columns]
    # Drop rows that are empty site labels
    df = df[keep].copy()
    df = df[df["Cancer"].notna()]
    # Basic numeric cleanup
    num_cols = [c for c in df.columns if c != "Cancer"]
    for c in num_cols:
        df[c] = (df[c].astype(str)
                      .str.replace(r"[^\d\.\-]", "", regex=True)
                      .replace({"": None}))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)

def parse_fact_sheet_table(pdf_path):
    """
    Parse the 'Incidence, Mortality and Prevalence by cancer site' table on page 2.
    Returns a tidy DataFrame or raises on failure.
    """
    # First try tabula (often returns multiple tables; choose the largest)
    dfs = tabula.read_pdf(str(pdf_path), pages=2, lattice=False, stream=True, guess=True)
    if dfs:
        candidate = max(dfs, key=lambda d: d.shape[0]*d.shape[1])
        out = _standardize_table(candidate)
        if len(out) >= 10 and "Cancer" in out.columns:
            return out

    # Fallback: pdfplumber – parse page 2 and try to capture the large table
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[1]  # 0-indexed; page 2 is index 1
        table = page.extract_table() or page.extract_tables()[0]
        df = pd.DataFrame(table[1:], columns=table[0])
        out = _standardize_table(df)
        return out

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
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))    # ← use your already-downloaded files

    print("Parsing page-2 tables… (this may take a few minutes)")
    rows = []
    for p in pdf_paths:
        meta = filename_to_meta(p)
        try:
            df = parse_fact_sheet_table(p)
            df.insert(0, "pop_code", meta["pop_code"])
            df.insert(1, "country", meta["country"])
            rows.append(df)
        except Exception as e:
            print("FAILED:", p, e)

    if not rows:
        raise SystemExit("No tables parsed. Check Java/Tabula install and try again.")

    big = pd.concat(rows, ignore_index=True)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    big.to_csv(OUT_DIR / "globocan2022_fact_sheets_page2.csv", index=False)
    big.to_parquet(OUT_DIR / "globocan2022_fact_sheets_page2.parquet", index=False)
    print(f"Done → {OUT_DIR/'globocan2022_fact_sheets_page2.csv'}")

if __name__ == "__main__":
    main()
