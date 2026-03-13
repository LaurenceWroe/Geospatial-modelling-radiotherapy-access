
#%% 
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


import pandas as pd
import pdfplumber
import re
from typing import List, Dict, Optional
import numpy as np
import fitz  # PyMuPDF - you'll need to install with: pip install PyMuPDF
import os
import glob
import numpy as np

# ---- Actual Script ---- #

import time, pathlib, concurrent.futures, tempfile
from pathlib import Path
from urllib.parse import urlparse
import requests


# --- selenium just to discover the PDF links reliably (page is JS-rendered) ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

#%% 

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

def main():
    print("Discovering fact-sheet PDFs…")
    links = list_population_factsheet_pdfs()
    print(f"Found {len(links)} PDFs")

    # Download politely, a few at a time
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(download_one, href) for _, href in links]
        pdf_paths = [f.result() for f in concurrent.futures.as_completed(futures)]
    pdf_paths.sort()

if __name__ == "__main__":
    main()

    #%% 

    def extract_cancer_table_from_pdf(pdf_path: str, save_cropped: bool = False, crop_factor: float = 0.6) -> pd.DataFrame:
        """
        Extract cancer statistics table from the second page of a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            save_cropped (bool): Whether to save the cropped page as a separate PDF
            
        Returns:
            pd.DataFrame: Extracted cancer statistics data
        """
    
        # Save cropped version if requested
        if save_cropped:
            try:
                out_dir = Path(pdf_path).parent / "crop"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{Path(pdf_path).stem}_cropped_page2.pdf"
                save_cropped_page_as_pdf(pdf_path, crop_factor=crop_factor, output_path=str(out_file))
            except ImportError:
                print("Warning: PyMuPDF not installed. Install with 'pip install PyMuPDF' to save cropped PDFs.")
            except Exception as e:
                print(f"Warning: Could not save cropped PDF: {e}")
        
        with pdfplumber.open(pdf_path) as pdf:
            # Get the second page (index 1)
            if len(pdf.pages) < 2:
                raise ValueError("PDF must have at least 2 pages")
            
            page = pdf.pages[1]  # Second page
            
            # Crop the page to focus on the table area (upper portion)
            # Adjust these coordinates based on your PDF layout
            # These values work for typical A4 pages - you may need to adjust
            page_height = page.height
            page_width = page.width
            
            # Crop to upper 60% of the page to exclude footer content
            cropped_page = page.crop((0, 0, page_width, page_height * crop_factor))
            
            # Try multiple extraction methods
            raw_table = None
            
            # Method 1: Try extract_tables first
            tables = cropped_page.extract_tables()
            if tables and len(tables) > 0:
                # Find the largest table (likely the main statistics table)
                largest_table = max(tables, key=lambda t: len(t) * len(t[0]) if t and t[0] else 0)
                if largest_table and len(largest_table) > 5:  # Should have multiple cancer types
                    raw_table = largest_table
            
            # Method 2: If no good table found, try text-based extraction
            if raw_table is None:
                raw_table = extract_table_from_text(cropped_page)
            
            if raw_table is None or len(raw_table) < 2:
                raise ValueError("Could not extract a valid table from the cropped page area")
            
            # Process the table
            df = process_cancer_table(raw_table)
            
            return df
    
def save_cropped_page_as_pdf(pdf_path: str, crop_factor: float, output_path: str = None):
    """
    Save the cropped second page as a separate PDF for visualization.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path for output PDF (optional, will auto-generate if None)
    """
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"globocan_factsheets/cropped/{base_name}_cropped_page2.pdf"

    # make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    if len(doc) < 2:
        raise ValueError("PDF must have at least 2 pages")
    
    # Get page 2 (index 1)
    page = doc[1]
    
    # Get page dimensions
    rect = page.rect
    page_height = rect.height
    page_width = rect.width
    
    # Create crop rectangle (upper 60% of page)
    crop_rect = fitz.Rect(0, 0, page_width, page_height * crop_factor)
    
    # Create new document with cropped page
    new_doc = fitz.open()  # Create empty document
    new_page = new_doc.new_page(width=crop_rect.width, height=crop_rect.height)
    
    # Copy the cropped area to the new page
    new_page.show_pdf_page(new_page.rect, doc, 1, clip=crop_rect)
    
    # Save the cropped PDF
    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    
    print(f"Cropped page saved to: {output_path}")
    return output_path

def extract_table_from_text(page) -> Optional[List[List[str]]]:
    """
    Fallback method to extract table data from text when table extraction fails.
    
    Args:
        page: pdfplumber page object
        
    Returns:
        List of lists representing table data, or None if extraction fails
    """
    text = page.extract_text()
    if not text:
        return None
    
    lines = text.split('\n')
    table_data = []
    
    # Look for lines that contain cancer data (typically have numbers)
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip obvious header/footer text
        if any(skip_word in line.lower() for skip_word in [
            'page', 'source', 'method', 'data source', 'incidence', 'mortality', 
            'prevalence', 'references', 'website', 'global cancer', 'cancer today'
        ]):
            continue
        
        # Look for lines with cancer names followed by numbers
        # Split on multiple spaces to separate columns
        parts = re.split(r'\s{2,}', line)
        
        # If we have multiple parts and at least one number, it's likely data
        if len(parts) >= 3 and any(re.search(r'\d+', part) for part in parts[1:]):
            table_data.append(parts)
    
    return table_data if len(table_data) > 5 else None
    
def process_cancer_table(raw_table: List[List[str]]) -> pd.DataFrame:   
    """
    Process the raw extracted table into a clean DataFrame.
    
    Args:
        raw_table: Raw table data as list of lists
        
    Returns:
        pd.DataFrame: Cleaned and structured cancer statistics
    """
    
    # Find the header row (usually contains "Cancer", "New cases", etc.)
    header_row_idx = None
    for i, row in enumerate(raw_table):
        if row and any(cell and 'Cancer' in str(cell) for cell in row):
            header_row_idx = i
            break
    
    if header_row_idx is None:
        # If no clear header found, assume first non-empty row is header
        for i, row in enumerate(raw_table):
            if any(cell and str(cell).strip() for cell in row):
                header_row_idx = i
                break
    
    # Extract headers and data
    headers = raw_table[header_row_idx] if header_row_idx is not None else raw_table[0]
    data_rows = raw_table[header_row_idx + 1:] if header_row_idx is not None else raw_table[1:]
    
    # Clean headers
    clean_headers = []
    for header in headers:
        if header:
            clean_headers.append(str(header).strip())
        else:
            clean_headers.append("")
    
    # Create column names based on the table structure
    # The table has: Cancer | New cases (Number, Rank, %, Cum risk) | Deaths (Number, Rank, %, Cum risk) | 5-year prevalence (Number, Prop per 100,000)
    
    column_names = [
        'Cancer',
        'New_Cases_Number', 'New_Cases_Rank', 'New_Cases_Percent', 'New_Cases_Cum_Risk',
        'Deaths_Number', 'Deaths_Rank', 'Deaths_Percent', 'Deaths_Cum_Risk',
        'Prevalence_Number', 'Prevalence_Prop_Per_100k'
    ]
    
    # Filter out empty rows and process data
    processed_data = []
    for row in data_rows:
        if not row or not any(cell and str(cell).strip() for cell in row):
            continue
            
        # Clean the row data
        clean_row = []
        for cell in row:
            if cell is None:
                clean_row.append(None)
            else:
                cell_str = str(cell).strip()
                # Handle numeric values
                if cell_str == '' or cell_str == '-':
                    clean_row.append(None)
                elif cell_str.replace('.', '').replace(',', '').isdigit():
                    # Convert numeric strings to appropriate type
                    try:
                        if '.' in cell_str:
                            clean_row.append(float(cell_str.replace(',', '')))
                        else:
                            clean_row.append(int(cell_str.replace(',', '')))
                    except ValueError:
                        clean_row.append(cell_str)
                else:
                    clean_row.append(cell_str)
        
        # Only add rows that have a cancer type (first column not empty)
        if clean_row and clean_row[0]:
            processed_data.append(clean_row)
    
    # Create DataFrame
    # Adjust column count to match data
    max_cols = max(len(row) for row in processed_data) if processed_data else len(column_names)
    
    # Adjust column names if needed
    if max_cols != len(column_names):
        column_names = column_names[:max_cols] + [f'Column_{i}' for i in range(len(column_names), max_cols)]
    
    # Pad rows to match column count
    for row in processed_data:
        while len(row) < max_cols:
            row.append(None)
        # Trim rows if they're too long
        row[:] = row[:max_cols]
    
    df = pd.DataFrame(processed_data, columns=column_names[:max_cols])
    
    # Clean up cancer names (remove any extra whitespace, handle special characters)
    if 'Cancer' in df.columns:
        df['Cancer'] = df['Cancer'].astype(str).str.strip()
    
    return df

def save_extracted_data(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """
    Save the extracted DataFrame to file.
    
    Args:
        df: DataFrame to save
        output_path: Path for output file
        format: Output format ('csv', 'excel', 'json')
    """
    
    if format.lower() == 'csv':
        df.to_csv(output_path, index=False)
    elif format.lower() == 'excel':
        df.to_excel(output_path, index=False)
    elif format.lower() == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError("Format must be 'csv', 'excel', or 'json'")

# Enhanced batch processing for country-specific PDFs using above
def process_country_cancer_pdfs(folder_path: str, output_dir: str = "globocan_processed_data", save_cropped: bool = False) -> pd.DataFrame:
    """
    Process multiple country cancer PDF files and combine into a single DataFrame.
    
    Args:
        folder_path: Path to folder containing PDF files named as (countrynumber)-(countryname)-fact-sheet.pdf
        output_dir: Directory to save output files
        save_cropped: Whether to save cropped versions of PDFs for debugging
        
    Returns:
        pd.DataFrame: Combined cancer statistics for all countries
    """
    

    # NEW: ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files matching the pattern (number-name-fact-sheet.pdf)
    pdf_pattern = os.path.join(folder_path, "*-*-fact-sheet.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        raise ValueError(f"No PDF files found matching pattern in {folder_path}")
    
    print(f"Found {len(pdf_files)} country PDF files to process")
    
    all_country_data = []
    processing_results = {}
    
    for pdf_path in pdf_files:
        try:
            # Extract country information from filename
            filename = os.path.basename(pdf_path)
            # Remove the '-fact-sheet.pdf' suffix and split
            country_part = filename.replace('-fact-sheet.pdf', '')
            parts = country_part.split('-', 1)  # Split on first dash only
            
            if len(parts) >= 2:
                try:
                    country_number = int(parts[0])  # Convert to integer
                    country_name = parts[1].replace('-', ' ').title()
                except ValueError:
                    # Fallback if first part isn't a number
                    country_number = None
                    country_name = country_part.replace('-', ' ').title()
            else:
                try:
                    country_number = int(parts[0])
                    country_name = f"Country_{country_number}"
                except ValueError:
                    country_number = None
                    country_name = parts[0].title()
            
            print(f"Processing {country_name} (#{country_number})...")
            
            # Extract the table data
            df = extract_cancer_table_from_pdf(pdf_path, save_cropped=save_cropped)
            
            # Add country information to the dataframe
            df['Country_Number'] = country_number
            df['Country_Name'] = country_name
            df['Source_File'] = filename
            
            # Reorder columns to put country info first
            cols = ['Country_Number', 'Country_Name'] + [col for col in df.columns if col not in ['Country_Number', 'Country_Name', 'Source_File']] + ['Source_File']
            df = df[cols]
            
            all_country_data.append(df)
            processing_results[pdf_path] = {
                'success': True, 
                'country_number': country_number,
                'country_name': country_name,
                'rows': len(df),
                'cancer_types': df['Cancer'].tolist() if 'Cancer' in df.columns else []
            }
            
            print(f"✓ Successfully processed {country_name} (#{country_number}): {len(df)} cancer types")
            
        except Exception as e:
            processing_results[pdf_path] = {
                'success': False, 
                'error': str(e),
                'country_number': None,
                'country_name': 'Unknown'
            }
            print(f"✗ Failed to process {pdf_path}: {str(e)}")
    
    if not all_country_data:
        raise ValueError("No PDF files were successfully processed")
    
    # Combine all dataframes
    combined_df = pd.concat(all_country_data, ignore_index=True)
    
    # Sort by country number for better organization
    # FIX: pandas uses na_position, not na_last
    combined_df = combined_df.sort_values(['Country_Number', 'Cancer'], na_position="last").reset_index(drop=True)

    # Save combined data
    combined_csv_path = os.path.join(output_dir, "all_countries_cancer_statistics.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    
    combined_excel_path = os.path.join(output_dir, "all_countries_cancer_statistics.xlsx")
    # NEW: be tolerant if openpyxl missing
    try:
        combined_df.to_excel(combined_excel_path, index=False)
    except Exception as e:
        print(f"Warning: couldn't write Excel ({e}). CSV was written: {combined_csv_path}")

    # Create summary report
    summary_report = create_processing_summary(processing_results, combined_df)
    summary_path = os.path.join(output_dir, "processing_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Countries processed: {combined_df['Country_Name'].nunique()}")
    print(f"Country numbers range: {combined_df['Country_Number'].min()}-{combined_df['Country_Number'].max()}")
    print(f"Total cancer type entries: {len(combined_df)}")
    print(f"Files saved:")
    print(f"  - Combined CSV: {combined_csv_path}")
    print(f"  - Combined Excel: {combined_excel_path}")
    print(f"  - Summary report: {summary_path}")
    
    return combined_df

def create_processing_summary(results: dict, combined_df: pd.DataFrame) -> str:
    """
    Create a summary report of the processing results.
    """
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    summary = f"CANCER PDF PROCESSING SUMMARY\n"
    summary += f"="*50 + "\n\n"
    summary += f"Total files processed: {len(results)}\n"
    summary += f"Successful: {successful}\n"
    summary += f"Failed: {failed}\n\n"
    
    if successful > 0:
        summary += f"COMBINED DATASET STATISTICS:\n"
        summary += f"- Total rows: {len(combined_df)}\n"
        summary += f"- Countries: {combined_df['Country_Name'].nunique()}\n"
        summary += f"- Country numbers range: {combined_df['Country_Number'].min()}-{combined_df['Country_Number'].max()}\n"
        summary += f"- Unique cancer types across all countries: {combined_df['Cancer'].nunique()}\n"
        summary += f"- Columns: {', '.join(combined_df.columns)}\n\n"
        
        summary += f"COUNTRIES SUCCESSFULLY PROCESSED (sorted by number):\n"
        successful_countries = []
        for file_path, result in results.items():
            if result['success']:
                successful_countries.append((result['country_number'], result['country_name'], result['rows']))
        
        # Sort by country number
        successful_countries.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))
        
        for country_number, country_name, rows in successful_countries:
            if country_number is not None:
                summary += f"- #{country_number:3d}: {country_name} ({rows} cancer types)\n"
            else:
                summary += f"- Unknown: {country_name} ({rows} cancer types)\n"
    
    if failed > 0:
        summary += f"\nFAILED FILES:\n"
        for file_path, result in results.items():
            if not result['success']:
                filename = os.path.basename(file_path)
                summary += f"- {filename}: {result['error']}\n"
    
    return summary

def build_country_tensor(
    combined_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    country_field: str = "Country_Name",
) -> tuple[np.ndarray, List[str], List[str], List[str]]:
    """
    Build a 3D tensor with shape (n_cancers, n_metrics, n_countries).
    Returns: (tensor, cancers, metrics, countries)
    """

    # Default metric list (only keep those present)
    default_metrics = [
        "New_Cases_Number", "New_Cases_Rank", "New_Cases_Percent", "New_Cases_Cum_Risk",
        "Deaths_Number", "Deaths_Rank", "Deaths_Percent", "Deaths_Cum_Risk",
        "Prevalence_Number", "Prevalence_Prop_Per_100k",
    ]
    if metrics is None:
        metrics = [m for m in default_metrics if m in combined_df.columns]

    # Ensure numeric (strip % and grouping separators where present)
    for m in metrics:
        if m in combined_df.columns:
            combined_df[m] = (
                combined_df[m]
                .astype(str)
                .str.replace("\u00A0", " ", regex=False)  # NBSP
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            combined_df[m] = pd.to_numeric(combined_df[m], errors="coerce")

    # Ensure we have ISO3 country codes to use as the country axis
    if "ISO3" not in combined_df.columns:
        if "Country_Code" in combined_df.columns:
            combined_df["ISO3"] = combined_df["Country_Code"].astype(str).str.upper()
        else:
            try:
                import pycountry
                def _to_iso3(name: str) -> Optional[str]:
                    try:
                        return pycountry.countries.lookup(str(name)).alpha_3.upper()
                    except Exception:
                        return None
                combined_df["ISO3"] = combined_df[country_field].apply(_to_iso3)
            except ImportError:
                raise ImportError(
                    "pycountry is required to derive ISO3 codes from names when 'Country_Code' is absent."
                )

    # Canonical cancer list (sorted)
    cancers = sorted(combined_df["Cancer"].dropna().astype(str).unique())

    # Canonical country list — prefer sorting by Country_Number if available, but return ISO3 codes
    if "Country_Number" in combined_df.columns:
        _countries = (
            combined_df[["Country_Number", "ISO3"]]
            .dropna(subset=["ISO3"])
            .drop_duplicates()
            )
        # fix: ensure numeric sort on Country_Number
        _countries["Country_Number"] = pd.to_numeric(_countries["Country_Number"], errors="coerce")
        _countries = _countries.sort_values(["Country_Number", "ISO3"], na_position="last")
        countries = _countries["ISO3"].tolist()
        
    else:
        countries = sorted(combined_df["ISO3"].dropna().astype(str).unique())

    # Allocate tensor
    tensor = np.full((len(cancers), len(metrics), len(countries)), np.nan, dtype=float)

    # Fill tensor by pivoting each metric
    for m_idx, m in enumerate(metrics):
        piv = combined_df.pivot_table(index="Cancer", columns="ISO3", values=m, aggfunc="first")
        piv = piv.reindex(index=cancers, columns=countries)  # align to canonical axes
        tensor[:, m_idx, :] = piv.to_numpy(dtype=float)

    return tensor, cancers, metrics, countries



# Example usage with debugging and batch processing
if __name__ == "__main__":
    
    # 1: Process a single PDF and save cropped version
    print("=== SINGLE PDF PROCESSING ===")
    pdf_path = "globocan_factsheets/pdf/4-afghanistan-fact-sheet.pdf" 

    if os.path.exists(pdf_path):
        try:
            # Extract the table and save cropped version for inspection
            cancer_data = extract_cancer_table_from_pdf(pdf_path, save_cropped=True)
            
            # Display basic info about the extracted data
            print(f"Extracted {len(cancer_data)} rows of cancer statistics")
            print(f"Columns: {list(cancer_data.columns)}")
            print("\nFirst few rows:")
            print(cancer_data.head(10))
            
            # Display some sample cancer types to verify extraction
            print(f"\nSample cancer types found:")
            if 'Cancer' in cancer_data.columns:
                print(cancer_data['Cancer'].head(10).tolist())
            
            # Save to CSV
            output_file = "afghan_cancer_statistics_extracted.csv"
            save_extracted_data(cancer_data, output_file, 'csv')
            print(f"\nData saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing single PDF: {str(e)}")
    else:
        print(f"File {pdf_path} not found for single processing example")
    
    print("\n" + "="*60)
    
    # Example 2: Process all country PDFs in a folder
    print("=== BATCH COUNTRY PROCESSING ===")
    folder_path = "globocan_factsheets/pdf"  # Replace with your folder path
    
    if os.path.exists(folder_path):
        try:
            # Process all country PDFs and combine into single dataset
            combined_data = process_country_cancer_pdfs(
                folder_path=folder_path,
                output_dir="globocan_processed_output",
                save_cropped=True
            )
            
            # Build 3D tensor: (cancer, metric, country)
            tensor, cancers, metrics, countries = build_country_tensor(combined_data)

            print(f"\n3D tensor shape:", tensor.shape)  # (n_cancers, n_metrics, n_countries)
            print("Example axes:", len(cancers), "cancers |", len(metrics), "metrics |", len(countries), "countries")

            # Save for later use
            import xarray as xr
            da = xr.DataArray(
                tensor,
                coords={"Cancer": cancers, "Metric": metrics, "ISO3": countries},
                dims=["Cancer", "Metric", "ISO3"],
            )

            da.to_netcdf("globocan_processed_output/globocan_xarray.nc")     # saves data + labels in one file

            print(f"\n=== COMBINED DATASET OVERVIEW ===")
            print(f"Shape: {combined_data.shape}")
            print(f"Countries: {sorted(combined_data['Country_Name'].unique().tolist())}")
            print(f"Sample of data:")
            
            print(combined_data[['Country_Name', 'Cancer', 'New_Cases_Number', 'Deaths_Number']].head(15))
            
            # --- helpers to identify aggregate rows robustly ---
            def _norm(s):
                s = str(s).lower().strip()
                s = re.sub(r'\s+', ' ', s)
                return s

            cnorm = combined_data['Cancer'].map(_norm)

            mask_all = cnorm.eq('all cancers')
            mask_all_excl = (
                cnorm.str.startswith('all cancers')
                & (cnorm.str.contains('excl') | cnorm.str.contains('excluding'))
                & (cnorm.str.contains('nmsc') | cnorm.str.contains('non-melanoma') | cnorm.str.contains('non melanoma'))
            )

            idx = ['Country_Number', 'Country_Name']

            # site-level rows only (exclude the two aggregates)
            sites_df = combined_data[~mask_all & ~mask_all_excl].copy()

            # count of site types per country (should be ~32)
            site_type_count = sites_df.groupby(idx)['Cancer'].nunique().rename('Cancer_Types_Count')

            # sum across site rows
            site_sums = sites_df.groupby(idx)[['New_Cases_Number', 'Deaths_Number']].sum()
            site_sums = site_sums.rename(columns={
                'New_Cases_Number': 'TopSitesSum_NewCases',
                'Deaths_Number': 'TopSitesSum_Deaths'
            })

            # country totals from "All cancers excl. NMSC" (preferred), fallback to "All cancers" if missing
            totals_excl = combined_data[mask_all_excl].groupby(idx)[['New_Cases_Number', 'Deaths_Number']].sum()
            totals_all  = combined_data[mask_all].groupby(idx)[['New_Cases_Number', 'Deaths_Number']].sum()

            all_keys = site_sums.index.union(totals_excl.index).union(totals_all.index)
            totals_excl = totals_excl.reindex(all_keys)
            totals_all  = totals_all.reindex(all_keys)
            totals = totals_excl.combine_first(totals_all)  # prefer excl-NMSC
            totals = totals.rename(columns={
                'New_Cases_Number': 'AllExclNMSC_NewCases',
                'Deaths_Number': 'AllExclNMSC_Deaths'
            })

            # residual = (All excl. NMSC) - (sum of top 32 site rows)
            summary = pd.concat([site_type_count, totals, site_sums], axis=1)

            # compute residuals and shares safely
            for a, b, r in [
                ('AllExclNMSC_NewCases', 'TopSitesSum_NewCases', 'Residual_Other_NewCases'),
                ('AllExclNMSC_Deaths',   'TopSitesSum_Deaths',   'Residual_Other_Deaths')
            ]:
                summary[r] = (summary[a] - summary[b]).clip(lower=0)

            summary['TopSitesShare_NewCases'] = (summary['TopSitesSum_NewCases'] / summary['AllExclNMSC_NewCases']).round(3)
            summary['TopSitesShare_Deaths']   = (summary['TopSitesSum_Deaths']   / summary['AllExclNMSC_Deaths']).round(3)

            # tidy and display
            summary = summary.sort_index()
            cols_order = [
                'Cancer_Types_Count',
                'AllExclNMSC_NewCases', 'TopSitesSum_NewCases', 'Residual_Other_NewCases', 'TopSitesShare_NewCases',
                'AllExclNMSC_Deaths',   'TopSitesSum_Deaths',   'Residual_Other_Deaths',   'TopSitesShare_Deaths',
            ]
            print("\n=== STATISTICS BY COUNTRY (cleaned) ===")
            print(summary[cols_order].head(15).round(0))
            
        except Exception as e:
            print(f"Error processing country PDFs: {str(e)}")
            
            # Try to provide some debug info
            pdf_files = glob.glob(os.path.join(folder_path, "*-*-fact-sheet.pdf"))
            if pdf_files:
                print(f"\nFound {len(pdf_files)} PDF files:")
                for pdf_file in pdf_files[:5]:  # Show first 5
                    print(f"  - {os.path.basename(pdf_file)}")
                if len(pdf_files) > 5:
                    print(f"  ... and {len(pdf_files) - 5} more")
            else:
                print(f"No files matching pattern '*-*-fact-sheet.pdf' found in {folder_path}")
    else:
        print(f"Folder {folder_path} not found")
        print("Please create the folder and add your country PDF files")
        print("Expected filename format: (countrynumber)-(countryname)-fact-sheet.pdf")
        print("Example: 4-Afghanistan-fact-sheet.pdf, 8-Albania-fact-sheet.pdf")
    
    print("\n" + "="*60)
    print("REQUIREMENTS:")
    print("- pip install pandas pdfplumber openpyxl")
    print("- pip install PyMuPDF  (for saving cropped PDFs)")
    print("\nFILE NAMING CONVENTION:")
    print("- (countrynumber)-(countryname)-fact-sheet.pdf")
    print("- Examples: 4-Afghanistan-fact-sheet.pdf, 8-Albania-fact-sheet.pdf, 840-United-States-fact-sheet.pdf")

#%%
# Inspecting the tensor

country = "AFG"
metric  = "New_Cases_Percent"
k = countries.index(country)
m = metrics.index(metric)

series_from_tensor = pd.Series(tensor[:, m, k], index=cancers).sort_values(ascending=False)
print(series_from_tensor.head(39))

#%%

print("tensor shape:", tensor.shape)
print("#cancers, #metrics, #countries =", len(cancers), len(metrics), len(countries))
print("sample cancers:", cancers[:5])
print("sample metrics:", metrics[:5])
print("sample countries (ISO3):", countries[:10])

