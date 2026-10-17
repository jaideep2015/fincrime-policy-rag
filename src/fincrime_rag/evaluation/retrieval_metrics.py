"""LLM-free retrieval metrics. Pure functions over ranked hits and gold (doc_id, pdf_page) pairs.

A hit *covers* a gold page when it has the same doc_id and its page list contains that page (a chunk
can span pages). Per question, with hits ranked best first:

* ``any@k``   1 if any of the top k hits covers at least one gold page (the usual "recall@k"),
* ``frac@k``  share of the gold pages covered by the top k hits (matters for multi-page / multi-doc),
* ``all@k``   1 if the top k hits cover *every* gold page (only differs from any@k when gold has 2+ pages),
* ``mrr``     1 / rank of the first hit covering a gold page (0 if none within the retrieved depth),
* ``hit``     1 if any of the top ``hit_k`` hits is from a gold *document* (page ignored): separates
              "found the right document" from "found the right page".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

Gold = frozenset  # frozenset[tuple[str, int]]


@dataclass(frozen=True)
class RankedHit:
    doc_id: str
    pdf_pages: tuple[int, ...]


def covered(hit: RankedHit, gold: Gold) -> set[tuple[str, int]]:
    return {(hit.doc_id, p) for p in hit.pdf_pages if (hit.doc_id, p) in gold}


def question_metrics(hits: list[RankedHit], gold: Gold, ks: list[int], hit_k: int) -> dict[str, float]:
    if not gold:
        raise ValueError("gold must be non-empty")
    gold_docs = {doc for doc, _ in gold}
    out: dict[str, float] = {}
    for k in ks:
        got: set[tuple[str, int]] = set()
        for hit in hits[:k]:
            got |= covered(hit, gold)
        out[f"any@{k}"] = 1.0 if got else 0.0
        out[f"frac@{k}"] = len(got) / len(gold)
        out[f"all@{k}"] = 1.0 if len(got) == len(gold) else 0.0
    out["mrr"] = next((1.0 / r for r, h in enumerate(hits, start=1) if covered(h, gold)), 0.0)
    out["hit"] = 1.0 if any(h.doc_id in gold_docs for h in hits[:hit_k]) else 0.0
    return out


def aggregate(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {name: sum(r[name] for r in rows) / len(rows) for name in rows[0]}


def summarize(
    per_question: list[tuple[str, dict[str, float]]],
) -> dict[str, tuple[int, dict[str, float]]]:
    """[(category, metrics)] -> {"overall": (n, means), "<category>": (n, means), ...}."""
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for category, metrics in per_question:
        groups["overall"].append(metrics)
        groups[category].append(metrics)
    ordered = ["overall"] + sorted(c for c in groups if c != "overall")
    return {name: (len(groups[name]), aggregate(groups[name])) for name in ordered}
