from fastapi import APIRouter, Depends, HTTPException
import time

from api.deps import get_engine
from api.schemas.search import SearchRequest
from src.agent.crag import CRAGWorkflow
from src.agent.llm_client import LLMClient
from src.agent.router import QueryRouter

router = APIRouter(prefix="/api/v1/agent", tags=["agentic"])

cross_encoder_instance = None

def get_agentic_services(engine = Depends(get_engine)):
    global cross_encoder_instance
    if cross_encoder_instance is None:
        from src.semantic.reranker import AgenticCrossEncoder
        cross_encoder_instance = AgenticCrossEncoder(engine.doc_store)
        
    crag = CRAGWorkflow(engine, cross_encoder_instance)
    llm = LLMClient()
    query_router = QueryRouter()
    return engine, crag, llm, query_router

@router.get("/smart")
def smart_search(
    params: SearchRequest = Depends(),
    services = Depends(get_agentic_services)
):
    engine, crag, llm, query_router = services
    
    start = time.time()
    
    # 1. Route the query
    route = query_router.route(params.q)
    
    # 2. Retrieve & Rerank via CRAG (Cross-Encoder)
    top_results = crag.evaluate_and_search(params.q, top_k=5, threshold=0.1)
    
    # Extract Context
    context = ""
    for doc_id, score in top_results:
        doc = engine.doc_store.get(doc_id)
        if doc:
            context += f"Source: {doc['title']}\n{doc['text'][:800]}\n\n"

    # 3. Generate Answer based on route
    if route == "compare":
        sys_prompt = "You are an expert at comparing subjects based on the provided context."
    elif route == "literature":
        sys_prompt = "You are an academic researcher. Synthesize a structured literature review based on the provided context."
    else:
        sys_prompt = "You are a helpful AI assistant. Answer the query concisely using ONLY the provided context. If the context doesn't contain the answer, say so."

    answer = llm.generate(sys_prompt, f"Context:\n{context}\n\nQuery: {params.q}")
    
    return {
        "query": params.q,
        "route": route,
        "answer": answer,
        "took_ms": round((time.time() - start) * 1000, 2),
        "sources": [{"doc_id": r[0], "score": r[1]} for r in top_results]
    }
