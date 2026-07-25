import json
import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphRetriever:
    def __init__(self, index_dir):
        self.graph = {"nodes": {}, "edges": []}
        graph_path = Path(index_dir) / "graph.json"
        if os.path.exists(graph_path):
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    self.graph = json.load(f)
                logger.info(f"Loaded knowledge graph with {len(self.graph['nodes'])} nodes.")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {e}")

    def score(self, query_tokens):
        """
        Given query tokens, find matching nodes.
        For each matching node, its documents get a score of 1.0.
        For documents connected via 1 hop (edges), they get a score of 0.5.
        Returns a dictionary of doc_id -> graph_score.
        """
        doc_scores = {}
        if not self.graph["nodes"]:
            return doc_scores
            
        matched_nodes = []
        for token in query_tokens:
            for node in self.graph["nodes"]:
                if token.lower() in node.lower():
                    matched_nodes.append(node)
                    
        # Direct matches
        for node in matched_nodes:
            docs = self.graph["nodes"][node].get("docs", [])
            for d in docs:
                doc_scores[str(d)] = doc_scores.get(str(d), 0.0) + 1.0
                
        # 1-hop expansion
        for edge in self.graph["edges"]:
            if edge["source"] in matched_nodes or edge["target"] in matched_nodes:
                d = str(edge.get("doc_id", ""))
                if d:
                    # Give 0.5 for 1-hop support
                    doc_scores[d] = doc_scores.get(d, 0.0) + 0.5
                    
        return doc_scores
