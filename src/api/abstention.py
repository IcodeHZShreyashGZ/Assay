"""Semantic Abstention Gate for Assay (Stage 4).

Inspects user natural language queries and evidence against demo/glossary.yaml:
  1. Ambiguity Detection: Catches under-specified terms (e.g. 'active users',
     'top customer') and returns structured clarification requests with concrete options.
  2. Unknown Metric Abstention: Detects complex ungrounded domain jargon
     (e.g. 'EBITDA', 'viral coefficient', 'burn multiplier') that cannot be computed
     from the schema, abstaining rather than hallucinating speculative SQL.
  3. Grounded Metric Enrichment: Binds known business metrics (e.g. 'ARPU', 'MRR',
     'churn rate') to certified SQL formulas and enriches domain evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

logger = logging.getLogger("assay.abstention")


@dataclass
class AbstentionDecision:
    """Outcome of semantic glossary evaluation."""
    should_abstain: bool
    reason: Optional[str] = None  # "AMBIGUOUS_TERM", "UNKNOWN_METRIC", "OUT_OF_SCHEMA", None
    clarification_message: Optional[str] = None
    options: List[str] = field(default_factory=list)
    matched_metrics: List[Dict[str, Any]] = field(default_factory=list)
    injected_evidence: Optional[str] = None


class SemanticAbstentionGate:
    """Evaluates NL queries against glossary.yaml rules and definitions."""

    def __init__(self, glossary_path: Optional[Union[str, Path]] = None):
        if glossary_path is None:
            # Default lookup paths
            candidates = [
                Path("demo/glossary.yaml"),
                Path(__file__).resolve().parent.parent.parent / "demo" / "glossary.yaml",
            ]
            for c in candidates:
                if c.exists():
                    self.glossary_path = c
                    break
            else:
                self.glossary_path = Path("demo/glossary.yaml")
        else:
            self.glossary_path = Path(glossary_path)

        self.glossary: Dict[str, Any] = self._load_glossary()

    def _load_glossary(self) -> Dict[str, Any]:
        """Load glossary YAML safely."""
        if not self.glossary_path.exists():
            logger.warning(f"Glossary file not found at {self.glossary_path}, using empty defaults.")
            return {"metrics": {}, "ambiguous_terms": {}, "abstention_rules": {}}

        try:
            with open(self.glossary_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data or {}
        except Exception as e:
            logger.error(f"Error loading glossary: {e}")
            return {"metrics": {}, "ambiguous_terms": {}, "abstention_rules": {}}

    def evaluate_query(
        self,
        question: str,
        evidence: str = "",
        available_tables: Optional[List[str]] = None,
    ) -> AbstentionDecision:
        """Evaluate query and evidence against enterprise business glossary."""
        q_norm = question.lower()
        ev_norm = evidence.lower() if evidence else ""
        combined = f"{q_norm} {ev_norm}"

        # 1. Check for Ambiguous Terms
        ambiguous_defs = self.glossary.get("ambiguous_terms", {})
        for term_key, term_info in ambiguous_defs.items():
            synonyms = term_info.get("synonyms", [term_key])
            for syn in synonyms:
                pattern = r"\b" + re.escape(syn.lower()) + r"\b"
                if re.search(pattern, q_norm):
                    # Check if evidence or question already resolved the ambiguity
                    resolved = False
                    for opt in term_info.get("options", []):
                        # Extract core keywords from options, e.g. "login", "purchase"
                        for word in re.findall(r"\b[a-zA-Z]{4,}\b", opt.lower()):
                            if word in ev_norm and word not in syn.lower():
                                resolved = True
                                break
                        if resolved:
                            break

                    if not resolved:
                        return AbstentionDecision(
                            should_abstain=True,
                            reason="AMBIGUOUS_TERM",
                            clarification_message=term_info.get("clarification", f"Term '{syn}' requires clarification."),
                            options=term_info.get("options", []),
                        )

        # 2. Check for Unknown / Unmapped Jargon (Enterprise Out-of-Schema Safety)
        # Typical financial / corporate terms that don't exist in standard databases
        known_unsupported_metrics = {
            "ebitda": "EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)",
            "burn rate": "Burn Rate (Monthly Cash Depletion)",
            "burn multiplier": "Burn Multiplier (Net Burn / Net New ARR)",
            "viral coefficient": "Viral Coefficient (K-Factor)",
            "magic number": "SaaS Magic Number",
            "rule of 40": "Rule of 40 Growth Index",
            "quick ratio": "SaaS Quick Ratio",
            "blended cac": "Blended Customer Acquisition Cost",
        }

        for jargon, full_name in known_unsupported_metrics.items():
            if re.search(r"\b" + re.escape(jargon) + r"\b", q_norm):
                # If neither the evidence nor the glossary defines this term
                metric_defined = False
                for m_info in self.glossary.get("metrics", {}).values():
                    if any(jargon in s.lower() for s in m_info.get("synonyms", [])):
                        metric_defined = True
                        break

                if not metric_defined:
                    return AbstentionDecision(
                        should_abstain=True,
                        reason="UNKNOWN_METRIC",
                        clarification_message=(
                            f"The metric '{full_name}' is not defined in the business glossary or database schema. "
                            f"Please define this business logic or provide relevant domain evidence before proceeding."
                        ),
                        options=[
                            "Provide custom SQL formula via domain evidence",
                            "Request enterprise administrator to add metric to glossary.yaml",
                        ],
                    )

        # 3. Check for Known Defined Metrics and Enrich Evidence
        matched_metrics = []
        injected_snippets = []

        metrics_dict = self.glossary.get("metrics", {})
        for m_key, m_info in metrics_dict.items():
            synonyms = m_info.get("synonyms", [m_key])
            for syn in synonyms:
                pattern = r"\b" + re.escape(syn.lower()) + r"\b"
                if re.search(pattern, q_norm):
                    matched_metrics.append(m_info)
                    name = m_info.get("name", m_key)
                    expr = m_info.get("sql_expression", "")
                    req_tables = ", ".join(m_info.get("required_tables", []))
                    snippet = f"Glossary Definition for {name}: formula `{expr}` (uses tables: {req_tables})"
                    if snippet not in injected_snippets:
                        injected_snippets.append(snippet)
                    break

        injected_evidence = None
        if injected_snippets:
            injected_evidence = "\n".join(injected_snippets)

        return AbstentionDecision(
            should_abstain=False,
            matched_metrics=matched_metrics,
            injected_evidence=injected_evidence,
        )
