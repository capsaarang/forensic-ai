"""
Forensic-AI — CLI entrypoint

Usage:
    python -m src.main --ticker AAPL --year 2023
    python -m src.main --file data/sample_10k/apple_2023.txt
    python -m src.main --ticker MSFT --year 2023 --focus risk_factors revenue litigation
    python -m src.main --ticker GS --year 2023 --focus all --output-dir outputs/
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from .analysis.focus_areas import FOCUS_AREAS
from .pipeline import run_pipeline, DEFAULT_FOCUS_AREAS


def main():
    parser = argparse.ArgumentParser(
        prog="forensic-ai",
        description="Forensic-AI: Agentic RAG audit pipeline for SEC 10-K filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --ticker AAPL --year 2023
  python -m src.main --ticker JPM  --year 2023 --focus risk_factors revenue debt litigation
  python -m src.main --file  data/sample_10k/apple_2023.txt --ticker AAPL --year 2023
  python -m src.main --ticker MSFT --year 2022 --focus all --no-agentic

Focus area keys:
  risk_factors     Item 1A — material risk disclosures
  revenue          MD&A — revenue recognition and anomalies
  debt             Liquidity, capital resources, debt schedule
  litigation       Item 3 and notes — legal proceedings
  related_party    Notes — related-party transactions
  forward_guidance MD&A outlook and forward-looking statements
  all              Run all six focus areas
        """,
    )

    # Source
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ticker", type=str, help="Company ticker symbol (e.g. AAPL)")
    source.add_argument("--file",   type=str, help="Path to local 10-K PDF or text file")

    parser.add_argument("--year",   type=int, default=2023, help="Fiscal year (default: 2023)")

    # Focus areas
    valid_keys = list(FOCUS_AREAS.keys()) + ["all"]
    parser.add_argument(
        "--focus",
        nargs="+",
        default=DEFAULT_FOCUS_AREAS,
        choices=valid_keys,
        metavar="AREA",
        help=f"Audit focus areas. One or more of: {', '.join(valid_keys)}",
    )

    # Pipeline config
    parser.add_argument("--chunk-size",  type=int, default=800,  help="Chunk size in chars (default: 800)")
    parser.add_argument("--overlap",     type=int, default=150,  help="Chunk overlap in chars (default: 150)")
    parser.add_argument("--k",           type=int, default=4,    help="FAISS results per query (default: 4)")
    parser.add_argument("--max-chunks",  type=int, default=8,    help="Max chunks per focus area (default: 8)")
    parser.add_argument("--no-agentic",  action="store_true",    help="Disable agentic follow-up retrieval")
    parser.add_argument("--output-dir",  type=str, default="outputs", help="Output directory (default: outputs/)")

    args = parser.parse_args()

    # Require at least ticker or file
    if not args.ticker and not args.file:
        parser.error("Provide either --ticker SYMBOL or --file PATH")

    # Expand 'all'
    focus_keys = list(FOCUS_AREAS.keys()) if "all" in args.focus else args.focus

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("  Export it:  export ANTHROPIC_API_KEY=your_key")
        print("  Or create a .env file — see .env.example")
        sys.exit(1)

    try:
        run_pipeline(
            ticker=args.ticker,
            year=args.year,
            file_path=args.file,
            focus_areas=focus_keys,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            k_per_query=args.k,
            max_chunks_per_focus=args.max_chunks,
            agentic_followup=not args.no_agentic,
        )
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
