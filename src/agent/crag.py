from src.agent.llm_client import LLMClient

class CRAGWorkflow:
    def __init__(self, engine, cross_encoder):
        self.engine = engine
        self.cross_encoder = cross_encoder
        self.llm = LLMClient()

    def evaluate_and_search(self, query, top_k=15, threshold=0.1):
        """
        Executes a search. If confidence is low, rewrites query and searches again.
        """
        # Fast BM25 + Cosine search
        results = self.engine.search(query, top_k)
        
        # Smart Cross-Encoder re-ranking
        reranked = self.cross_encoder.rerank(query, results)

        # Check confidence
        is_confident = False
        if reranked and reranked[0][1] > threshold:
            is_confident = True
            
        if is_confident:
            return reranked

        # Corrective Action: Rewrite Query
        rewrite_prompt = """
        You are an expert query rewriter for an information retrieval system.
        The user's query did not return highly relevant results. 
        Rewrite the query using broader terms, synonyms, or related concepts to improve search recall.
        Do not include quotes or conversational text. Respond ONLY with the new query text.
        """
        new_query = self.llm.generate(rewrite_prompt, f"Original Query: {query}").strip()
        
        # Fallback to original if LLM fails or is empty
        if not new_query or "Error" in new_query:
            return reranked

        print(f"CRAG Triggered! Rewrote '{query}' -> '{new_query}'")

        # Search again with the rewritten query
        results2 = self.engine.search(new_query, top_k)
        reranked2 = self.cross_encoder.rerank(new_query, results2)
        
        return reranked2
