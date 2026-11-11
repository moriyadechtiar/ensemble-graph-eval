"""Semantic similarity judge. Needs the optional ``sentence-transformers`` package."""
from typing import Any, Dict

import numpy as np

from .graphs import extract_edges, extract_nodes, normalize_graph_schema, safe_json_parse

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(name: str = DEFAULT_EMBEDDING_MODEL):
    """Load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def cosine_similarity(a, b):
    """Cosine similarity between the rows of two 2-D arrays."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


class SemanticSimilarity:
    """Semantic Similarity - Embedding-based comparison."""

    def __init__(self, embedding_model):
        self.name = "semantic_similarity"
        self.model = embedding_model

    def _graph_to_text(self, graph: dict) -> str:
        """
        Convert graph to text representation.
        """
        if not isinstance(graph, dict):
            return ""

        parts = []

        nodes = extract_nodes(graph)
        for node in nodes:
            node_type = node.get('node_type', node.get('type', ''))
            node_id = node.get('id', '')

            attr = ''
            attrs = node.get('attrs', {})
            for attr_name in ['name', 'text', 'title', 'term']:
                if attr_name in node and node[attr_name]:
                    attr = node[attr_name]
                    break
                elif isinstance(attrs, dict) and attr_name in attrs and attrs[attr_name]:
                    attr = attrs[attr_name]
                    break

            parts.append(f"{node_type} {node_id} {attr}")

        # Add edges
        edges = extract_edges(graph)
        for edge in edges:
            src = edge.get('src', edge.get('source', ''))
            tgt = edge.get('tgt', edge.get('target', ''))
            edge_type = edge.get('type', edge.get('edge_type', ''))
            parts.append(f"{src} {edge_type} {tgt}")

        return " ".join(parts)

    def score(self, pred: dict, gold: dict) -> Dict[str, Any]:
        pred_json = safe_json_parse(pred) if isinstance(pred, str) else pred
        gold_json = safe_json_parse(gold) if isinstance(gold, str) else gold

        if pred_json is None or gold_json is None:
            return {
                'score': 0.0,
                'cosine_similarity': 0.0,
                'valid_json': pred_json is not None
            }

        pred_json = normalize_graph_schema(pred_json)
        gold_json = normalize_graph_schema(gold_json)

        # Convert to text
        pred_text = self._graph_to_text(pred_json)
        gold_text = self._graph_to_text(gold_json)

        if not pred_text or not gold_text:
            return {
                'score': 0.0,
                'cosine_similarity': 0.0,
                'valid_json': True
            }

        # Compute embeddings
        pred_emb = self.model.encode([pred_text])
        gold_emb = self.model.encode([gold_text])

        # Compute cosine similarity
        similarity = float(cosine_similarity(pred_emb, gold_emb)[0][0])

        return {
            'score': similarity,
            'cosine_similarity': similarity,
            'valid_json': True
        }
