"""Reference-based judges. Each compares a predicted graph with a reference graph.

Every judge has a ``name`` and a ``score(pred, gold)`` method that returns a
dictionary whose ``score`` entry is in [0, 1], higher is better. ``pred`` and
``gold`` can be graph dictionaries or JSON text.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from .graphs import (calculate_prf1, extract_edges, extract_nodes, normalize_graph_schema,
                     normalize_node_for_comparison, normalize_text, safe_json_parse)


class ExactMatch:
    """Exact Match - Binary exact JSON comparison."""

    def __init__(self):
        self.name = "exact_match"

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        pred_json = safe_json_parse(pred) if isinstance(pred, str) else pred
        gold_json = safe_json_parse(gold) if isinstance(gold, str) else gold

        if pred_json is None or gold_json is None:
            return {
                'score': 0.0,
                'exact_match': False,
                'valid_json': pred_json is not None
            }

        # Normalize for comparison
        pred_str = json.dumps(pred_json, sort_keys=True)
        gold_str = json.dumps(gold_json, sort_keys=True)

        exact_match = pred_str == gold_str

        return {
            'score': 1.0 if exact_match else 0.0,
            'exact_match': exact_match,
            'valid_json': True
        }

class StructuralF1:
    """Structural F1 - Node and edge matching with F1 scores."""

    def __init__(self):
        self.name = "structural_f1"

    def _node_to_tuple(self, node: dict) -> tuple:
        """
        Convert node to comparable tuple with schema-aware attribute extraction.
        """
        node_id = node.get('id', '')
        node_type = node.get('node_type', node.get('type', '')).upper()

        key_attr = ''
        attrs = node.get('attrs', {})

        for attr_name in ['name', 'term', 'text', 'title']:
            if attr_name in node and node[attr_name]:
                key_attr = normalize_text(str(node[attr_name]))
                break
            elif isinstance(attrs, dict) and attr_name in attrs and attrs[attr_name]:
                key_attr = normalize_text(str(attrs[attr_name]))
                break

        if not key_attr and node_id:
            key_attr = normalize_text(str(node_id))

        return (node_type, key_attr)

    def _edge_to_tuple(self, edge: dict, node_map: dict) -> tuple:
        """
        Convert edge to comparable tuple with fallback for unmapped nodes.
        """
        src = edge.get('src', edge.get('source', edge.get('from', '')))
        tgt = edge.get('tgt', edge.get('target', edge.get('to', '')))
        edge_type = edge.get('type', edge.get('edge_type', edge.get('label', ''))).upper()

        src_key = node_map.get(str(src), str(src)) if src else ''
        tgt_key = node_map.get(str(tgt), str(tgt)) if tgt else ''

        return (src_key, edge_type, tgt_key)

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        pred_json = safe_json_parse(pred) if isinstance(pred, str) else pred
        gold_json = safe_json_parse(gold) if isinstance(gold, str) else gold

        if pred_json is None or gold_json is None:
            return {
                'score': 0.0,
                'node_precision': 0.0,
                'node_recall': 0.0,
                'node_f1': 0.0,
                'edge_precision': 0.0,
                'edge_recall': 0.0,
                'edge_f1': 0.0,
                'valid_json': pred_json is not None
            }

        pred_json = normalize_graph_schema(pred_json)
        gold_json = normalize_graph_schema(gold_json)

        pred_nodes = extract_nodes(pred_json)
        gold_nodes = extract_nodes(gold_json)
        pred_edges = extract_edges(pred_json)
        gold_edges = extract_edges(gold_json)

        pred_node_map = {str(n.get('id', '')): self._node_to_tuple(n) for n in pred_nodes}
        gold_node_map = {str(n.get('id', '')): self._node_to_tuple(n) for n in gold_nodes}

        pred_node_set = set(pred_node_map.values())
        gold_node_set = set(gold_node_map.values())

        pred_edge_set = {self._edge_to_tuple(e, pred_node_map) for e in pred_edges}
        gold_edge_set = {self._edge_to_tuple(e, gold_node_map) for e in gold_edges}

        node_tp = len(pred_node_set & gold_node_set)
        node_fp = len(pred_node_set - gold_node_set)
        node_fn = len(gold_node_set - pred_node_set)

        node_p, node_r, node_f1 = calculate_prf1(node_tp, node_fp, node_fn)

        edge_tp = len(pred_edge_set & gold_edge_set)
        edge_fp = len(pred_edge_set - gold_edge_set)
        edge_fn = len(gold_edge_set - pred_edge_set)

        edge_p, edge_r, edge_f1 = calculate_prf1(edge_tp, edge_fp, edge_fn)

        # Overall score (average of node and edge F1)
        overall_score = (node_f1 + edge_f1) / 2.0

        return {
            'score': overall_score,
            'node_precision': node_p,
            'node_recall': node_r,
            'node_f1': node_f1,
            'edge_precision': edge_p,
            'edge_recall': edge_r,
            'edge_f1': edge_f1,
            'valid_json': True
        }

class ParseSuccess:
    """Parse Success - JSON validity and structure checks."""

    def __init__(self):
        self.name = "parse_success"

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        pred_json = safe_json_parse(pred) if isinstance(pred, str) else pred

        if pred_json is None:
            return {
                'score': 0.0,
                'valid_json': False,
                'has_nodes': False,
                'has_edges': False,
                'has_contract_id': False
            }
        pred_json = normalize_graph_schema(pred_json)
        has_nodes = 'nodes' in pred_json and len(pred_json.get('nodes', [])) > 0
        has_edges = 'edges' in pred_json and len(pred_json.get('edges', [])) > 0
        has_contract_id = 'contract_id' in pred_json

        # Score based on presence of expected fields
        score = 0.0
        if has_contract_id:
            score += 0.2
        if has_nodes:
            score += 0.4
        if has_edges:
            score += 0.4

        return {
            'score': score,
            'valid_json': True,
            'has_nodes': has_nodes,
            'has_edges': has_edges,
            'has_contract_id': has_contract_id
        }

class Completeness:
    """Completeness - Coverage and recall-focused metrics."""
    def __init__(self):
        self.name = "completeness"

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        pred_json = safe_json_parse(pred) if isinstance(pred, str) else pred
        gold_json = safe_json_parse(gold) if isinstance(gold, str) else gold

        if pred_json is None or gold_json is None:
            return {
                'score': 0.0,
                'node_coverage': 0.0,
                'edge_coverage': 0.0,
                'type_coverage': 0.0,
                'valid_json': pred_json is not None
            }

        pred_json = normalize_graph_schema(pred_json)
        gold_json = normalize_graph_schema(gold_json)
        pred_nodes = extract_nodes(pred_json)
        gold_nodes = extract_nodes(gold_json)
        pred_edges = extract_edges(pred_json)
        gold_edges = extract_edges(gold_json)

        # Node coverage (how many gold nodes are covered)
        gold_node_ids = {n.get('id') for n in gold_nodes}
        pred_node_ids = {n.get('id') for n in pred_nodes}
        node_coverage = (len(gold_node_ids & pred_node_ids) / max(1, len(gold_node_ids))) if gold_node_ids else 0.0

        # Edge coverage (match by (source,target,type))
        def edge_key(e):
            return (
                e.get('source') or e.get('from'),
                e.get('target') or e.get('to'),
                e.get('type') or e.get('label') or e.get('relation')
            )
        gold_edge_keys = {edge_key(e) for e in gold_edges}
        pred_edge_keys = {edge_key(e) for e in pred_edges}
        edge_coverage = (len(gold_edge_keys & pred_edge_keys) / max(1, len(gold_edge_keys))) if gold_edge_keys else 0.0

        # Type coverage: fraction of gold node types present in pred
        def node_type(n):
            return n.get('type') or n.get('label') or n.get('category')
        gold_types = {node_type(n) for n in gold_nodes if node_type(n)}
        pred_types = {node_type(n) for n in pred_nodes if node_type(n)}
        type_coverage = (len(gold_types & pred_types) / max(1, len(gold_types))) if gold_types else 0.0

        score = 0.5 * node_coverage + 0.3 * edge_coverage + 0.2 * type_coverage
        return {
            'score': float(score),
            'node_coverage': float(node_coverage),
            'edge_coverage': float(edge_coverage),
            'type_coverage': float(type_coverage),
            'valid_json': True,
            'num_gold_nodes': len(gold_nodes),
            'num_pred_nodes': len(pred_nodes),
            'num_gold_edges': len(gold_edges),
            'num_pred_edges': len(pred_edges)
        }

class GraphEditDistance:
    """Graph Edit Distance - Measures minimum edit operations to transform pred to gold."""

    def __init__(self):
        self.name = "graph_edit_distance"

    def _normalize_node(self, node: dict) -> tuple:
        """Convert node to normalized tuple for comparison."""
        node_id = str(node.get('id', ''))
        node_type = str(node.get('node_type', node.get('type', ''))).upper()

        attr = ''
        attrs = node.get('attrs', {})
        for key in ['name', 'text', 'title', 'term']:
            if key in node and node[key]:
                attr = normalize_text(str(node[key]))
                break
            elif isinstance(attrs, dict) and key in attrs and attrs[key]:
                attr = normalize_text(str(attrs[key]))
                break

        return (node_id, node_type, attr)

    def _normalize_edge(self, edge: dict) -> tuple:
        """Convert edge to normalized tuple for comparison."""
        src = str(edge.get('src', edge.get('source', edge.get('from', ''))))
        tgt = str(edge.get('tgt', edge.get('target', edge.get('to', ''))))
        edge_type = str(edge.get('type', edge.get('edge_type', edge.get('label', '')))).upper()

        return (src, tgt, edge_type)

    def _calculate_node_distance(self, node1: tuple, node2: tuple) -> int:
        """Calculate edit distance between two nodes."""
        id1, type1, attr1 = node1
        id2, type2, attr2 = node2

        distance = 0

        if id1 != id2:
            distance += 1

        if type1 != type2:
            distance += 1

        if attr1 != attr2:
            distance += 1

        return distance

    def _approximate_ged(self, pred_nodes: list, gold_nodes: list,
                         pred_edges: list, gold_edges: list) -> Dict[str, Any]:
        """Calculate approximate Graph Edit Distance using greedy matching."""

        pred_nodes_norm = [self._normalize_node(n) for n in pred_nodes]
        gold_nodes_norm = [self._normalize_node(n) for n in gold_nodes]
        pred_edges_norm = [self._normalize_edge(e) for e in pred_edges]
        gold_edges_norm = [self._normalize_edge(e) for e in gold_edges]

        pred_id_map = {n[0]: n for n in pred_nodes_norm}
        gold_id_map = {n[0]: n for n in gold_nodes_norm}

        node_edits = 0
        matched_gold_nodes = set()

        for pred_node in pred_nodes_norm:
            best_match = None
            best_distance = float('inf')

            for i, gold_node in enumerate(gold_nodes_norm):
                if i in matched_gold_nodes:
                    continue

                distance = self._calculate_node_distance(pred_node, gold_node)

                if distance < best_distance:
                    best_distance = distance
                    best_match = i

            if best_match is not None:
                matched_gold_nodes.add(best_match)
                node_edits += best_distance
            else:
                node_edits += 3

        node_insertions = len(gold_nodes_norm) - len(matched_gold_nodes)
        node_edits += node_insertions * 3

        pred_edge_set = set(pred_edges_norm)
        gold_edge_set = set(gold_edges_norm)

        edge_deletions = len(pred_edge_set - gold_edge_set)
        edge_insertions = len(gold_edge_set - pred_edge_set)
        edge_edits = edge_deletions + edge_insertions

        total_ged = node_edits + edge_edits

        max_nodes = max(len(pred_nodes_norm), len(gold_nodes_norm), 1)
        max_edges = max(len(pred_edges_norm), len(gold_edges_norm), 1)
        max_distance = (max_nodes * 3) + max_edges

        normalized_distance = min(1.0, total_ged / max_distance) if max_distance > 0 else 0.0

        similarity_score = max(0.0, min(1.0, 1.0 - normalized_distance))

        return {
            'ged': total_ged,
            'node_edits': node_edits,
            'edge_edits': edge_edits,
            'node_insertions': node_insertions,
            'node_deletions': len(pred_nodes_norm) - len(matched_gold_nodes),
            'edge_insertions': edge_insertions,
            'edge_deletions': edge_deletions,
            'normalized_distance': normalized_distance,
            'similarity_score': similarity_score
        }

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        pred_json = safe_json_parse(pred) if isinstance(pred, str) else pred
        gold_json = safe_json_parse(gold) if isinstance(gold, str) else gold

        if pred_json is None or gold_json is None:
            return {
                'score': 0.0,
                'ged': None,
                'valid_json': pred_json is not None
            }

        pred_json = normalize_graph_schema(pred_json)
        gold_json = normalize_graph_schema(gold_json)

        pred_nodes = extract_nodes(pred_json)
        gold_nodes = extract_nodes(gold_json)
        pred_edges = extract_edges(pred_json)
        gold_edges = extract_edges(gold_json)

        ged_result = self._approximate_ged(pred_nodes, gold_nodes, pred_edges, gold_edges)

        return {
            'score': ged_result['similarity_score'],
            'ged': ged_result['ged'],
            'node_edits': ged_result['node_edits'],
            'edge_edits': ged_result['edge_edits'],
            'node_insertions': ged_result['node_insertions'],
            'node_deletions': ged_result['node_deletions'],
            'edge_insertions': ged_result['edge_insertions'],
            'edge_deletions': ged_result['edge_deletions'],
            'normalized_distance': ged_result['normalized_distance'],
            'valid_json': True
        }

class Hybrid:
    """Hybrid - Combined structural and semantic scoring."""

    def __init__(self, structural_judge, semantic_judge, alpha: float = 0.5):
        self.name = "hybrid"
        self.structural = structural_judge
        self.semantic = semantic_judge
        self.alpha = alpha

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        struct_result = self.structural.score(pred, gold)
        sem_result = self.semantic.score(pred, gold)

        struct_score = struct_result.get('score', 0.0)
        sem_score = sem_result.get('score', 0.0)

        # Weighted combination
        hybrid_score = self.alpha * struct_score + (1 - self.alpha) * sem_score

        return {
            'score': hybrid_score,
            'structural_score': struct_score,
            'semantic_score': sem_score,
            'alpha': self.alpha,
            'valid_json': struct_result.get('valid_json', False)
        }
