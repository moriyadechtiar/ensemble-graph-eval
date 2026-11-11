"""Reference-free checks: score a predicted graph on its own, without a reference graph.

Each check returns a value in [0, 1], higher is better.
"""
import re
from collections import defaultdict
from typing import Any, Dict, Optional

EDGE_RULES = {
    "IS_PART_OF": ("CLAUSE", "CLAUSE"), "REFERENCES": ("CLAUSE", "CLAUSE"),
    "DEFINES": ("CLAUSE", "DEFINED_TERM"), "USES": ("CLAUSE", "DEFINED_TERM"),
    "MENTIONS_PARTY": ("CLAUSE", "PARTY"), "CONTAINS": ("CLAUSE", "VALUE"),
}
CHECK_NAMES = ("unused_definitions", "undefined_references", "duplicate_definitions", "orphan_nodes",
               "isolated_components", "domain_range_validity", "ontology_compliance")
SMALL_COMPONENT_MAX_NODES = 8


def _term(x: Any) -> str:
    return re.sub(r'["\u201c\u201d]', "", str(x if x is not None else "")).strip().lower()


def reference_free_scores(graph: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Score a graph dictionary. A missing or unparseable graph scores 0.0 on every check."""
    if not isinstance(graph, dict):
        return {name: 0.0 for name in CHECK_NAMES}
    if "nodes" not in graph and isinstance(graph.get("graph"), dict):
        graph = graph["graph"]
    nodes, edges = list(graph.get("nodes") or []), list(graph.get("edges") or [])
    ntype = {str(n.get("id")): str(n.get("node_type", n.get("type", "")) or "").upper() for n in nodes}
    src = lambda e: str(e.get("src") or e.get("source") or e.get("from"))
    tgt = lambda e: str(e.get("tgt") or e.get("target") or e.get("to"))
    etype = lambda e: str(e.get("type", e.get("edge_type", "")) or "").upper()
    out = {}

    terms = [i for i, t in ntype.items() if t == "DEFINED_TERM"]
    defined = {tgt(e) for e in edges if etype(e) == "DEFINES"}
    out["unused_definitions"] = 1.0 - sum(i not in defined for i in terms) / max(1, len(terms))

    targets = {tgt(e) for e in edges if etype(e) == "REFERENCES"}
    out["undefined_references"] = 1.0 - sum(ntype.get(t) != "CLAUSE" for t in targets) / max(1, len(targets))

    by_name = defaultdict(list)
    for n in nodes:
        name = _term(n.get("name") or n.get("label") or "")
        if ntype.get(str(n.get("id"))) == "DEFINED_TERM" and name:
            by_name[name].append(n.get("id"))
    out["duplicate_definitions"] = 1.0 - sum(len(v) > 1 for v in by_name.values()) / max(1, len(terms))

    connected = {src(e) for e in edges} | {tgt(e) for e in edges}
    out["orphan_nodes"] = 1.0 - len(set(ntype) - connected) / max(1, len(ntype))

    adj = defaultdict(set)
    for e in edges:
        adj[src(e)].add(tgt(e)); adj[tgt(e)].add(src(e))
    seen, components = set(), []
    for i in ntype:
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            u = stack.pop(); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        components.append(comp)
    out["isolated_components"] = 1.0 if len(components) <= 1 else \
        1.0 - sum(len(c) <= SMALL_COMPONENT_MAX_NODES for c in components[1:]) / max(1, len(components) - 1)

    typed = [e for e in edges if ntype.get(src(e)) and ntype.get(tgt(e))]
    invalid = sum(1 for e in typed if EDGE_RULES.get(etype(e)) != (ntype[src(e)], ntype[tgt(e)]))
    out["domain_range_validity"] = 1.0 - invalid / max(1, len(edges))

    bad, part_of = 0, set()
    for e in edges:
        rule = EDGE_RULES.get(etype(e))
        if ntype.get(src(e)) is None or ntype.get(tgt(e)) is None or rule is None or (ntype[src(e)], ntype[tgt(e)]) != rule:
            bad += 1
        if etype(e) == "IS_PART_OF":
            part_of.add((src(e), tgt(e)))
    with_parent = no_parent = no_edge = 0
    for i, t in ntype.items():
        if t != "CLAUSE" or "." not in i:
            continue
        with_parent += 1
        prefix = i.rsplit(".", 1)[0]
        parents = [p for p in (prefix, f"ARTICLE {prefix}") if ntype.get(p) == "CLAUSE"]
        if not parents:
            no_parent += 1
        elif not any((i, p) in part_of for p in parents):
            no_edge += 1
    out["ontology_compliance"] = max(0.0, min(1.0, 1.0 - (bad + no_parent + no_edge) / max(1, len(edges) + 2 * with_parent)))
    return out
