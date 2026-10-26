"""Turn-level signals of anchoring that the harness can read without the agent's cooperation."""

from __future__ import annotations

import re

# Phrases an agent uses when it declares the search finished. The paper's VLIW single agent
# spent 66% of its budget on turns that read <= 1 file, made no edit, and said things like
# "I've reached the practical limit of this machine configuration".
PLATEAU_PATTERNS = (
    r"practical limit",
    r"theoretical (?:limit|minimum|maximum|optimum)",
    r"diminishing returns",
    r"\b(?:already|essentially|effectively) (?:near[- ]?)?optimal",
    r"\bno (?:further|more) (?:meaningful |significant |obvious )?"
    r"(?:improvement|optimi[sz]ation|gain)s?\b",
    r"\b(?:reached|hit|hitting|at) (?:a|the) (?:plateau|ceiling)",
    r"\b(?:cannot|can't|can not) be (?:improved|optimi[sz]ed) (?:any )?further",
    r"\bnothing (?:more|else) to (?:improve|optimi[sz]e)",
)
_PLATEAU = re.compile("|".join(PLATEAU_PATTERNS), re.IGNORECASE)


def is_plateau_claim(text: str) -> bool:
    """True when the agent's end-of-turn message declares a plateau or a limit."""
    return bool(text) and _PLATEAU.search(text) is not None
