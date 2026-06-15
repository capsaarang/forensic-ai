"""
Forensic-AI Pipeline

Orchestrates the full audit pipeline:
  1. Load 10-K (from EDGAR or local file)
  2. Detect sections
  3. Chunk into retrieval units
  4. Embed chunks → FAISS vector store
  5. Retrieve top chunks per focus area
  6. LLM audit analysis (with agentic follow-up)
  7. Score + format output
  8. Save report
"""

from .ingestion.loader import load_from_file, load_from_edgar
from .ingestion.section_detector import detect_sections
from .ingestion.chunker import chunk_sections
from .retrieval.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .retrieval.retriever import Retriever
from .analysis.auditor import Auditor
from .analysis.focus_areas import FOCUS_AREAS
from .output.report import save_report, print_summary


DEFAULT_FOCUS_AREAS = ["risk_factors", "revenue", "debt", "litigation"]


def run_pipeline(
    ticker: str | None = None,
    year: int | None = None,
    file_path: str | None = None,
    focus_areas: list[str] | None = None,
    output_dir: str = "outputs",
    chunk_size: int = 800,
    overlap: int = 150,
    k_per_query: int = 4,
    max_chunks_per_focus: int = 8,
    agentic_followup: bool = True,
    api_key: str | None = None,
) -> dict:
    """
    Run the full Forensic-AI audit pipeline.

    Args:
        ticker:              Company ticker (e.g. 'AAPL') — used with EDGAR fetch
        year:                Fiscal year (e.g. 2023) — used with EDGAR fetch
        file_path:           Path to local PDF or text file (overrides EDGAR)
        focus_areas:         List of focus area keys to audit
        output_dir:          Directory for saving reports
        chunk_size:          Target characters per chunk
        overlap:             Chunk overlap in characters
        k_per_query:         FAISS results per query
        max_chunks_per_focus: Max chunks passed to LLM per focus area
        agentic_followup:    Enable LLM-driven follow-up retrieval
        api_key:             Anthropic API key (falls back to env var)

    Returns:
        Dict with 'result' (AuditResult) and 'paths' (saved file paths)
    """
    fiscal_year = str(year) if year else "unknown"
    focus_keys = focus_areas or DEFAULT_FOCUS_AREAS

    # Validate focus areas
    for key in focus_keys:
        if key not in FOCUS_AREAS:
            raise ValueError(f"Unknown focus area: '{key}'. Valid: {list(FOCUS_AREAS.keys())}")

    print(f"\n{'='*60}")
    print(f"  FORENSIC-AI  |  {(ticker or 'LOCAL FILE').upper()}  |  FY{fiscal_year}")
    print(f"  Focus areas: {', '.join(focus_keys)}")
    print(f"{'='*60}\n")

    # ── STAGE 1: Load document ───────────────────────────────────
    print("[1/6] Loading 10-K document...")
    if file_path:
        raw_text = load_from_file(file_path)
        ticker = ticker or "UNKNOWN"
    elif ticker and year:
        raw_text = load_from_edgar(ticker, year)
    else:
        raise ValueError("Provide either --file or both --ticker and --year")
    print(f"      {len(raw_text):,} characters loaded")

    # ── STAGE 2: Section detection ───────────────────────────────
    print("\n[2/6] Detecting 10-K sections...")
    sections = detect_sections(raw_text)
    print(f"      {len(sections)} sections identified:")
    for s in sections:
        print(f"        · {s.item_number}: {s.name} ({len(s.text):,} chars)")

    # ── STAGE 3: Chunking ────────────────────────────────────────
    print(f"\n[3/6] Chunking sections (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_sections(sections, chunk_size=chunk_size, overlap=overlap)
    print(f"      {len(chunks)} chunks created")

    # ── STAGE 4: Embedding + vector store ───────────────────────
    print("\n[4/6] Embedding chunks into FAISS vector store...")
    embedder = Embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_chunks(texts)

    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embeddings)
    print(f"      Vector store ready: {store}")

    # ── STAGE 5: RAG retrieval ───────────────────────────────────
    print(f"\n[5/6] Retrieving relevant chunks for {len(focus_keys)} focus areas...")
    retriever = Retriever(embedder, store)
    retrieved = retriever.retrieve(
        focus_keys=focus_keys,
        k_per_query=k_per_query,
        max_chunks_per_focus=max_chunks_per_focus,
    )

    # ── STAGE 6: LLM audit analysis ─────────────────────────────
    print(f"\n[6/6] Running LLM audit analysis (agentic={agentic_followup})...")
    auditor = Auditor(api_key=api_key)
    result = auditor.run(
        ticker=ticker,
        fiscal_year=fiscal_year,
        retrieved_context=retrieved,
        retriever=retriever if agentic_followup else None,
        total_chunks=len(chunks),
    )

    # ── OUTPUT ───────────────────────────────────────────────────
    print_summary(result)
    paths = save_report(result, output_dir=output_dir)

    return {"result": result, "paths": paths}
