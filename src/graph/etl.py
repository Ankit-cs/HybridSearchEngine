import json
import os
import re
from src.agent.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphETL:
    def __init__(self):
        self.llm = LLMClient()
        self.system_prompt = (
            "You are a Knowledge Graph extraction agent. Given a text chunk, extract entities and their relationships. "
            "Return ONLY a valid JSON list of dictionaries with keys: 'source', 'target', and 'relation'. "
            "Ensure the entities are concise (1-3 words). Do not include markdown formatting or extra text."
        )

    def extract_relations(self, text):
        user_prompt = f"Text to extract from:\n{text}\n\nOutput JSON only:"
        response = self.llm.generate(self.system_prompt, user_prompt)
        
        try:
            # Strip potential markdown formatting (e.g., ```json ... ```)
            cleaned = re.sub(r'^```json\s*|\s*```$', '', response.strip())
            relations = json.loads(cleaned)
            
            # Validate format
            valid_relations = []
            for r in relations:
                if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r:
                    valid_relations.append({
                        "source": str(r["source"]).strip(),
                        "target": str(r["target"]).strip(),
                        "relation": str(r["relation"]).strip()
                    })
            return valid_relations
        except Exception as e:
            logger.error(f"Failed to parse LLM ETL response: {e}\nResponse was: {response}")
            return []

    def merge_graphs(self, existing_graph, new_relations, doc_id):
        """
        existing_graph format:
        {
            "nodes": {"Entity1": {"docs": ["1", "2"]}, "Entity2": {"docs": ["1"]}},
            "edges": [{"source": "Entity1", "target": "Entity2", "relation": "KNOWS", "doc_id": "1"}]
        }
        """
        for rel in new_relations:
            src = rel["source"]
            tgt = rel["target"]
            relation = rel["relation"]
            
            # Add nodes
            if src not in existing_graph["nodes"]:
                existing_graph["nodes"][src] = {"docs": []}
            if str(doc_id) not in existing_graph["nodes"][src]["docs"]:
                existing_graph["nodes"][src]["docs"].append(str(doc_id))
                
            if tgt not in existing_graph["nodes"]:
                existing_graph["nodes"][tgt] = {"docs": []}
            if str(doc_id) not in existing_graph["nodes"][tgt]["docs"]:
                existing_graph["nodes"][tgt]["docs"].append(str(doc_id))
                
            # Add edge
            existing_graph["edges"].append({
                "source": src,
                "target": tgt,
                "relation": relation,
                "doc_id": str(doc_id)
            })
        
        return existing_graph
