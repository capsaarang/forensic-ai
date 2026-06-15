# Forensic-AI — Architecture

## Overview

Forensic-AI is an agentic Retrieval-Augmented Generation (RAG) pipeline designed to audit SEC 10-K annual filings. It automates the first pass of financial document review — surfacing anomalies, risk signals, and disclosure gaps that a human auditor or analyst would investigate.

---

## Pipeline Stages

### Stage 1 — Document Ingestion

**Module:** `src/ingestion/loader.py`

Two ingestion paths:

| Path | How |
|---|---|
| SEC EDGAR | Resolves ticker → CIK via EDGAR submissions API, fetches primary 10-K document, strips HTML |
| Local file | Reads PDF via `pdfplumber` or plain text directly |

Output: raw UTF-8 text string of the full filing (~100,000–500,000 characters for a typical 10-K).

---

### Stage 2 — Section Detection

**Module:** `src/ingestion/section_detector.py`

Uses regex matching against known SEC Item headers (Item 1, Item 1A, Item 7, Item 8, etc.) to split the document into labeled sections. Each `Section` carries:

- `item_number` (e.g. "Item 1A")
- `name` (e.g. "Risk Factors")
- `text` (raw section content)
- `start_char` / `end_char` (position in original document)

A separate `extract_notes()` function identifies financial statement notes (Note 1, Note 14, etc.) which frequently contain related-party and contingency disclosures.

---

### Stage 3 — Semantic Chunking

**Module:** `src/ingestion/chunker.py`

Each section is split into overlapping chunks with paragraph-aware boundaries:

- Default chunk size: **800 characters**
- Default overlap: **150 characters**
- Very long paragraphs (>1,200 chars) are further split at sentence boundaries

Overlap preserves context at chunk boundaries, preventing retrieval from missing relevant text split across two chunks.

Each `Chunk` carries its `section_name`, `item_number`, and position metadata.

---

### Stage 4 — Embedding + Vector Store

**Modules:** `src/retrieval/embedder.py`, `src/retrieval/vector_store.py`

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers)
- 384-dimensional dense vectors
- Embeddings are L2-normalized, enabling cosine similarity via dot product

**Vector store:** FAISS `IndexFlatIP` (exact inner product search)
- All chunk embeddings are indexed in memory
- Supports batch search across multiple query vectors simultaneously

A typical 10-K produces 40–200 chunks, well within FAISS flat index performance range.

---

### Stage 5 — RAG Retrieval

**Module:** `src/retrieval/retriever.py`

For each selected audit focus area, a set of domain-specific semantic queries is embedded and searched against the FAISS index:

```
Focus Area: "Revenue Anomalies"
  Query 1: "revenue recognition policy changes"
  Query 2: "deferred revenue contract liabilities backlog"
  Query 3: "non-GAAP adjusted revenue reconciliation"
  ...
```

Results across queries are deduplicated (same chunk may match multiple queries — best score wins), then sorted and truncated to `max_chunks_per_focus` (default 8).

This produces a focused context window per focus area — only the most semantically relevant passages, not the whole document.

---

### Stage 6 — LLM Audit Analysis

**Module:** `src/analysis/auditor.py`

For each focus area, a detailed prompt is constructed with:
1. Audit instructions specific to that focus area (what to look for)
2. Retrieved chunks as context
3. A structured JSON output specification

Claude is asked to produce 2–4 findings per focus area as a JSON array.

#### Agentic Follow-Up Loop

If the LLM determines it needs more context (signaled by `FOLLOWUP_REQUEST: <query>`), the retriever fetches additional chunks on demand. This loop runs up to `MAX_FOLLOWUP_ROUNDS = 2` times per focus area.

This is the "agentic" part: the LLM drives its own context gathering rather than passively consuming a fixed context window.

After all focus areas are analyzed, a final pass generates an executive summary and overall risk score (0–100).

---

### Stage 7 — Output

**Modules:** `src/output/formatter.py`, `src/output/report.py`

Two output formats:

| Format | Use case |
|---|---|
| **JSON** | Machine-readable, structured findings. Suitable for downstream storage (Snowflake), dashboards, or API responses |
| **Markdown** | Human-readable audit report with severity badges, flagged text, and recommendations. Suitable for sharing with stakeholders |

---

## Focus Areas

| Key | Label | Primary 10-K Sections |
|---|---|---|
| `risk_factors` | Risk Factors | Item 1A |
| `revenue` | Revenue Anomalies | Item 7 (MD&A) |
| `debt` | Debt & Liquidity | Item 7, Notes |
| `litigation` | Litigation Exposure | Item 3, Notes |
| `related_party` | Related-Party Transactions | Notes (esp. Note 14+) |
| `forward_guidance` | Forward Guidance | Item 7 Outlook |

Each focus area has 6–8 retrieval queries and a set of audit instructions that guide the LLM on what to flag.

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           INPUT SOURCES                 │
                    │   SEC EDGAR API  │  Local PDF/Text      │
                    └────────┬─────────────────┬──────────────┘
                             │                 │
                             ▼                 ▼
                    ┌─────────────────────────────────────────┐
                    │           loader.py                     │
                    │   CIK lookup → fetch filing → raw text  │
                    └────────────────┬────────────────────────┘
                                     │  raw text (str)
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │         section_detector.py             │
                    │   Item 1A / Item 7 / Item 3 / Notes...  │
                    └────────────────┬────────────────────────┘
                                     │  List[Section]
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │            chunker.py                   │
                    │   Paragraph-aware overlapping chunks    │
                    └────────────────┬────────────────────────┘
                                     │  List[Chunk]
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │    embedder.py + vector_store.py        │
                    │   all-MiniLM-L6-v2 → FAISS IndexFlatIP │
                    └────────────────┬────────────────────────┘
                                     │  FAISS index (in memory)
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │            retriever.py                 │
                    │   Focus-area queries → top-k chunks     │
                    └────────────────┬────────────────────────┘
                                     │  Dict[focus_key → List[Chunk]]
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │             auditor.py                  │
                    │   Claude prompt → JSON findings         │
                    │   ┌─────────────────────────────────┐  │
                    │   │  Agentic follow-up loop (×2)    │  │
                    │   │  LLM requests → retriever fetch  │  │
                    │   └─────────────────────────────────┘  │
                    └────────────────┬────────────────────────┘
                                     │  AuditResult
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │         formatter.py + report.py        │
                    │   JSON findings  │  Markdown report     │
                    └─────────────────────────────────────────┘
```

---

## Technology Choices

### Why sentence-transformers over OpenAI embeddings?
- Runs locally, no additional API cost per audit
- `all-MiniLM-L6-v2` is fast and well-benchmarked for semantic similarity
- Swappable: any HuggingFace embedding model works as a drop-in

### Why FAISS over a hosted vector DB?
- A single 10-K produces 40–200 chunks — FAISS flat index is instant at this scale
- No infrastructure dependency: runs entirely in-process
- Roadmap: Pinecone or pgvector for multi-filing, persistent storage

### Why Anthropic Claude for the LLM?
- Long context window handles multi-chunk prompts cleanly
- Strong instruction-following for structured JSON output
- Reliable refusal to hallucinate when told to cite only provided context

---

## Roadmap

### Phase 2 — Persistence
- Snowflake: store findings, scores, and chunk metadata per audit run
- Enable YoY delta analysis (compare findings across fiscal years)

### Phase 3 — Infrastructure
- AWS Lambda: serverless pipeline triggered by EDGAR filing RSS feed
- S3: store raw filings and output reports

### Phase 4 — Product
- Streamlit dashboard: interactive finding explorer with section viewer
- Multi-filing batch audit (entire portfolio of companies)
- 10-Q support (quarterly filings)
- Fine-tuned embedding model on SEC financial language (FinBERT variants)
