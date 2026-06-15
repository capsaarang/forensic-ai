"""
Risk Scorer

Computes a quantitative risk score (0-100) from audit findings.
Used as a cross-check against the LLM-generated score.
"""

SEVERITY_WEIGHTS = {
    "HIGH":   20,
    "MEDIUM": 8,
    "LOW":    3,
    "INFO":   0,
}

FOCUS_AREA_WEIGHTS = {
    "risk_factors":     1.3,
    "revenue":          1.4,
    "debt":             1.2,
    "litigation":       1.1,
    "related_party":    1.5,   # highest weight — most likely to indicate fraud
    "forward_guidance": 0.9,
}


def compute_heuristic_score(findings: list) -> int:
    """
    Compute a heuristic risk score from findings.

    Args:
        findings: List of Finding objects

    Returns:
        Integer score 0–100
    """
    if not findings:
        return 0

    raw = 0.0
    for f in findings:
        weight = SEVERITY_WEIGHTS.get(f.severity, 0)
        area_mult = FOCUS_AREA_WEIGHTS.get(f.focus_area, 1.0)
        raw += weight * area_mult

    # Normalize: a filing with 5 HIGH findings across key areas ≈ 80
    max_expected = 5 * 20 * 1.4
    score = min(100, int((raw / max_expected) * 100))
    return score


def severity_breakdown(findings: list) -> dict[str, int]:
    """Count findings by severity."""
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    for f in findings:
        sev = f.severity.upper()
        if sev in counts:
            counts[sev] += 1
    return counts


def top_findings(findings: list, n: int = 3) -> list:
    """Return the n highest-severity findings for the executive summary."""
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    return sorted(findings, key=lambda f: order.get(f.severity, 3))[:n]
