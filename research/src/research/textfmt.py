"""Plain-text formatting shared by the command-line reports."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["fmt_num", "fmt_pct", "fmt_share", "format_table"]


def fmt_num(value: float | None) -> str:
    """Four significant digits (no trailing zeros), whole numbers with separators from 1,000."""
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def fmt_pct(value: float | None) -> str:
    """Signed percentage with one decimal; "-" when undefined."""
    return "-" if value is None else f"{value:+.1f}%"


def fmt_share(value: float | None) -> str:
    """A fraction as a whole percentage ("66%"); "-" when undefined."""
    return "-" if value is None else f"{100 * value:.0f}%"


def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], *, align: str | None = None
) -> str:
    """Plain-text table. `align` is one 'l'/'r' per column (default: first left, rest right)."""
    align = align or "l" + "r" * (len(headers) - 1)
    widths = [max([len(h), *(len(r[i]) for r in rows)]) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        parts = (
            cell.ljust(width) if how == "l" else cell.rjust(width)
            for cell, width, how in zip(cells, widths, align, strict=True)
        )
        return "  ".join(parts).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(headers), rule, *(line(r) for r in rows)])
