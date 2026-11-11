"""Graph parsing and normalisation helpers shared by the judges."""
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def normalize_node_for_comparison(node: dict) -> dict:
    """
    Normalize node structure to handle both schemas:
    1. Flat: {"id": "x", "node_type": "PARTY", "name": "ACME"}
    2. Nested: {"id": "x", "type": "PARTY", "attrs": {"name": "ACME"}}

    This flattens the structure and ensures consistent field names.
    """
    if not isinstance(node, dict):
        return node

    normalized = dict(node)

    # If attrs exists, merge it to top level (don't overwrite existing)
    if 'attrs' in node and isinstance(node['attrs'], dict):
        attrs = node['attrs']
        for key, value in attrs.items():
            if key not in normalized:
                normalized[key] = value

    # Ensure both 'type' and 'node_type' are available at top level
    if 'type' in normalized and 'node_type' not in normalized:
        normalized['node_type'] = normalized['type']
    elif 'node_type' in normalized and 'type' not in normalized:
        normalized['type'] = normalized['node_type']

    return normalized


def normalize_graph_schema(graph: dict) -> dict:
    """Normalize entire graph schema to handle different formats."""
    if not isinstance(graph, dict):
        return graph

    normalized = dict(graph)

    # Normalize nodes
    if 'nodes' in graph and isinstance(graph['nodes'], list):
        normalized['nodes'] = [normalize_node_for_comparison(n) for n in graph['nodes']]

    return normalized


def extract_nodes(graph: dict) -> List[dict]:
    """Extract nodes from graph."""
    return graph.get('nodes', []) if isinstance(graph, dict) else []


def extract_edges(graph: dict) -> List[dict]:
    """Extract edges from graph."""
    return graph.get('edges', []) if isinstance(graph, dict) else []


def safe_json_parse(text: str) -> Optional[dict]:
    """Safely parse JSON from text."""
    if not isinstance(text, str):
        return None

    try:
        return json.loads(text)
    except:
        # Try to extract JSON from text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except:
                pass
    return None


def calculate_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1
