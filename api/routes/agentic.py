from fastapi import APIRouter, Depends, HTTPException
import time

from api.deps import get_engine
from api.schemas.search import SearchRequest
from src.agent.crag import CRAGWorkflow
from src.agent.llm_client import LLMClient
from src.agent.router import QueryRouter
from src.agent.memory import WorkingMemoryBuffer, EpisodicMemoryStore

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

    route = query_router.route(params.q)

    top_results = crag.evaluate_and_search(params.q, top_k=5, threshold=0.1)

    context = ""
    for doc_id, score in top_results:
        doc = engine.doc_store.get(doc_id)
        if doc:
            context += f"Source: {doc['title']}\n{doc['text'][:800]}\n\n"

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


@router.post("/memory/remember")
def remember(
    body: dict,
    engine = Depends(get_engine)
):
    agent_id = body.get("agent_id", "default")
    text = body.get("text", "")
    importance = body.get("importance", 1.0)
    session_id = body.get("session_id", "")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    store = engine.agent_partitions.get_partition(agent_id)
    entry = store.add(
        text=text,
        importance=importance,
        session_id=session_id,
    )
    return {
        "entry_id": entry.entry_id,
        "agent_id": agent_id,
        "recency_weight": entry.recency_weight,
        "total_memories": store.count(),
    }


@router.post("/memory/recall")
def recall(
    body: dict,
    engine = Depends(get_engine)
):
    from src.semantic.embedding_model import EmbeddingModel
    import numpy as np

    agent_id = body.get("agent_id", "default")
    query = body.get("query", "")
    top_k = body.get("top_k", 5)

    store = engine.agent_partitions.get_partition(agent_id)
    if store.count() == 0:
        return {"results": [], "total_memories": 0}

    model = EmbeddingModel()
    query_emb = model.encode(query)[0]
    results = store.search(query_emb, top_k=top_k)

    return {
        "results": [
            {
                "entry_id": e.entry_id,
                "text": e.text,
                "importance": e.importance_score,
                "recency_weight": e.recency_weight,
                "access_count": e.access_count,
            }
            for e in results
        ],
        "total_memories": store.count(),
    }


@router.post("/memory/decay")
def decay_memories(
    body: dict,
    engine = Depends(get_engine)
):
    agent_id = body.get("agent_id", "default")
    store = engine.agent_partitions.get_partition(agent_id)
    updated = store.decay_all()
    return {
        "agent_id": agent_id,
        "entries_updated": updated,
        "total_entries": store.count(),
    }


@router.get("/memory/stats")
def memory_stats(
    agent_id: str = "default",
    engine = Depends(get_engine)
):
    store = engine.agent_partitions.get_partition(agent_id)
    return {
        "agent_id": agent_id,
        "stats": store.get_stats(),
    }


@router.post("/memory/working/push")
def push_working_memory(
    body: dict,
    engine = Depends(get_engine)
):
    text = body.get("text", "")
    importance = body.get("importance", 1.0)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    engine.working_memory.push(text=text, importance=importance)
    return {
        "buffer_size": engine.working_memory.size(),
        "is_full": engine.working_memory.is_full(),
    }


@router.post("/memory/working/drain")
def drain_working_memory(
    engine = Depends(get_engine)
):
    items = engine.working_memory.drain()
    return {
        "drained_count": len(items),
        "items": [{"text": i["text"][:200], "importance": i["importance"]} for i in items[:10]],
    }


@router.get("/catalog/snapshots")
def list_snapshots(
    engine = Depends(get_engine)
):
    return {"snapshots": engine.get_time_travel_versions()}


@router.post("/catalog/snapshot")
def create_snapshot(
    body: dict,
    engine = Depends(get_engine)
):
    description = body.get("description", "Manual snapshot")
    result = engine.create_snapshot(description)
    return result


@router.post("/catalog/restore")
def restore_snapshot(
    body: dict,
    engine = Depends(get_engine)
):
    version = body.get("version")
    if version is None:
        raise HTTPException(status_code=400, detail="Version is required")
    success = engine.restore_version(version)
    return {"success": success, "version": version}


@router.get("/catalog/stats")
def catalog_stats(
    engine = Depends(get_engine)
):
    return engine.get_catalog_stats()


@router.get("/catalog/schema")
def current_schema(
    engine = Depends(get_engine)
):
    return {"schema": engine.schema_evolver.get_current_columns()}


@router.post("/catalog/schema/add-column")
def add_column(
    body: dict,
    engine = Depends(get_engine)
):
    column_name = body.get("column_name", "")
    column_type = body.get("column_type", "str")
    description = body.get("description", "")
    if not column_name:
        raise HTTPException(status_code=400, detail="column_name is required")
    schema = engine.schema_evolver.add_column(column_name, column_type, description)
    return {"schema": schema}
