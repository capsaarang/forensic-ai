"""
10-K Document Loader

Supports:
  1. Local PDF files (via pdfplumber)
  2. Local plain-text files
  3. SEC EDGAR full-text search API (fetch by ticker + year)
"""

import re
import time
import requests

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={year}-01-01&enddt={year}-12-31&forms=10-K"
EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=10-K&dateb=&owner=include&count=10&search_text="
EDGAR_HEADERS = {"User-Agent": "ForensicAI research@example.com"}  # SEC requires a user agent


def load_from_file(path: str) -> str:
    """
    Load a 10-K document from a local file (PDF or text).

    Args:
        path: Path to .pdf or .txt file

    Returns:
        Raw text content of the filing
    """
    if path.lower().endswith(".pdf"):
        return _load_pdf(path)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def load_from_edgar(ticker: str, year: int) -> str:
    """
    Fetch a 10-K filing from SEC EDGAR by ticker and fiscal year.

    Uses the EDGAR full-text search API to locate the filing,
    then fetches the primary document.

    Args:
        ticker: Company ticker symbol (e.g. 'AAPL')
        year: Fiscal year (e.g. 2023)

    Returns:
        Raw text content of the 10-K filing
    """
    print(f"[EDGAR] Searching for {ticker} 10-K filed around {year}...")

    # Step 1: Get CIK from ticker
    cik = _get_cik(ticker)
    if not cik:
        raise ValueError(f"Could not find CIK for ticker: {ticker}")
    print(f"[EDGAR] Found CIK: {cik}")

    # Step 2: Get list of 10-K filings
    filings = _get_10k_filings(cik)
    if not filings:
        raise ValueError(f"No 10-K filings found for {ticker} (CIK: {cik})")

    # Step 3: Find filing closest to target year
    target_filing = _select_filing_by_year(filings, year)
    if not target_filing:
        raise ValueError(f"No 10-K found for {ticker} around year {year}")

    print(f"[EDGAR] Selected filing: {target_filing['filingDate']} — {target_filing['primaryDocument']}")

    # Step 4: Fetch the filing document
    accession = target_filing["accessionNumber"].replace("-", "")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{target_filing['primaryDocument']}"
    print(f"[EDGAR] Fetching: {doc_url}")

    text = _fetch_filing_text(doc_url)
    print(f"[EDGAR] Retrieved {len(text):,} characters")
    return text


def _get_cik(ticker: str) -> str | None:
    """Resolve ticker to SEC CIK number."""
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type=10-K&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        # Try the company facts API as fallback
        facts_url = f"https://data.sec.gov/submissions/CIK{ticker.upper().zfill(10)}.json"
        resp2 = requests.get(facts_url, headers=EDGAR_HEADERS, timeout=15)
        if resp2.status_code == 200:
            data = resp2.json()
            return str(data.get("cik", "")).lstrip("0")
    except Exception:
        pass

    # Try ticker→CIK mapping
    try:
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"])
    except Exception:
        pass

    return None


def _get_10k_filings(cik: str) -> list[dict]:
    """Fetch list of 10-K filings from EDGAR submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form == "10-K":
            filings.append({
                "filingDate": dates[i],
                "accessionNumber": accessions[i],
                "primaryDocument": primary_docs[i],
            })

    return filings


def _select_filing_by_year(filings: list[dict], year: int) -> dict | None:
    """Pick the 10-K filing whose date is closest to December of target year."""
    target = f"{year}-12-31"
    best = None
    best_diff = float("inf")
    for f in filings:
        date_str = f["filingDate"]
        # Accept filings from Oct year to Mar year+1 (typical 10-K window)
        filing_year = int(date_str[:4])
        if filing_year in (year, year + 1):
            diff = abs(_date_to_days(date_str) - _date_to_days(target))
            if diff < best_diff:
                best_diff = diff
                best = f
    return best


def _date_to_days(date_str: str) -> int:
    """Convert YYYY-MM-DD to an integer for comparison."""
    parts = date_str.split("-")
    return int(parts[0]) * 365 + int(parts[1]) * 30 + int(parts[2])


def _fetch_filing_text(url: str) -> str:
    """Download and extract text from an EDGAR filing document."""
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    content = resp.text

    # Strip HTML tags if present
    if "<html" in content.lower() or "<body" in content.lower():
        content = _strip_html(content)

    return content


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    if not PDF_AVAILABLE:
        raise ImportError("pdfplumber is required for PDF loading. Run: pip install pdfplumber")

    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts)
