"""
Report Writer

Saves audit results to disk as JSON and/or Markdown.
"""

import os
from .formatter import to_json, to_markdown


def save_report(audit_result, output_dir: str = "outputs") -> dict[str, str]:
    """
    Save audit result to JSON and Markdown files.

    Args:
        audit_result: AuditResult object
        output_dir: Directory to write files into (created if not exists)

    Returns:
        Dict with 'json' and 'markdown' keys pointing to saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    base = f"{audit_result.ticker}_{audit_result.fiscal_year}_{audit_result.audit_date}"
    json_path = os.path.join(output_dir, f"{base}.json")
    md_path   = os.path.join(output_dir, f"{base}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(to_json(audit_result))

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(audit_result))

    print(f"[Report] Saved JSON  → {json_path}")
    print(f"[Report] Saved MD    → {md_path}")

    return {"json": json_path, "markdown": md_path}


def print_summary(audit_result) -> None:
    """Print a compact terminal summary using rich if available, else plain text."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()
        r = audit_result

        console.print()
        console.print(Panel(
            f"[bold]{r.ticker}[/bold] · FY{r.fiscal_year} · Risk Score: [bold red]{r.risk_score}/100[/bold red]",
            title="[bold blue]Forensic-AI Audit Complete[/bold blue]",
            subtitle=f"Model: {r.model_used}",
        ))

        console.print(f"\n[italic]{r.summary}[/italic]\n")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Sev", width=8)
        table.add_column("Section", width=22)
        table.add_column("Finding", width=44)
        table.add_column("Focus Area", width=18)

        sev_colors = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "cyan", "INFO": "white"}

        for f in r.findings:
            color = sev_colors.get(f.severity, "white")
            table.add_row(
                f"[{color}]{f.severity}[/{color}]",
                f.section[:20],
                f.title,
                f.focus_area.replace("_", " ").title(),
            )

        console.print(table)

    except ImportError:
        # Fallback: plain text
        r = audit_result
        print(f"\n{'='*60}")
        print(f"FORENSIC-AI AUDIT COMPLETE")
        print(f"  Ticker:     {r.ticker}")
        print(f"  FY:         {r.fiscal_year}")
        print(f"  Risk Score: {r.risk_score}/100")
        print(f"  Findings:   {len(r.findings)}")
        print(f"\nSummary: {r.summary}")
        print(f"\nFindings:")
        for f in r.findings:
            print(f"  [{f.severity}] {f.title} ({f.section})")
        print("="*60)
