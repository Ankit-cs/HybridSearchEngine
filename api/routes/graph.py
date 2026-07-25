import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from src.utils.config import INDEX_DIR

router = APIRouter(prefix="/api/v1", tags=["graph"])

@router.get("/graph")
def get_graph_data():
    graph_path = Path(INDEX_DIR) / "graph.json"
    if not os.path.exists(graph_path):
        return {"nodes": [], "edges": []}
        
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Transform for react-force-graph format:
        # nodes: [{ id, name, val }]
        # edges: [{ source, target, name }]
        
        nodes = []
        for node_id, node_data in data.get("nodes", {}).items():
            doc_count = len(node_data.get("docs", []))
            nodes.append({
                "id": node_id,
                "name": node_id,
                "val": max(1, doc_count) # node size
            })
            
        edges = []
        for edge in data.get("edges", []):
            edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "name": edge["relation"]
            })
            
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
