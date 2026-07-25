from fastapi import APIRouter, Depends, HTTPException, Query
import time
from src.utils.snippet import generate_snippet
from src.query.distributed import scatter_gather_search

from api.deps import get_engine
from api.schemas.search import SearchRequest, SearchResponse, SearchResult


router = APIRouter(prefix="/api/v1", tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search(
    params: SearchRequest = Depends(),
    engine = Depends(get_engine)
):
    start = time.time()

    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    search_kwargs = {
        "agent_id": params.agent_id,
        "column_filter": params.column_filter,
        "use_fts": params.use_fts,
        "use_dual": params.use_dual,
        "profile": params.profile
    }

    if params.profile:
        results, profile_data = engine.search(params.q, params.k, **search_kwargs)
    else:
        results = engine.search(params.q, params.k, **search_kwargs)
        profile_data = None

    response_results = []
    for doc_id, score in results:
        doc = engine.doc_store.get(doc_id)
        if not doc:
            continue
        snippet = generate_snippet(doc["text"], params.q)
        response_results.append(
            SearchResult(
                doc_id=doc_id,
                title=doc["title"],
                url=doc["url"],
                snippet=snippet,
                score=score,
            )
        )

    return SearchResponse(
        query=params.q,
        k=params.k,
        took_ms=round((time.time() - start) * 1000, 2),
        results=response_results,
        profile_data=profile_data
    )


@router.get("/search/explain")
def search_explain(
    params: SearchRequest = Depends(),
    engine = Depends(get_engine)
):
    start = time.time()
    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = engine.explain_search(
        params.q, top_k=params.k, agent_id=params.agent_id
    )
    
    # Generate snippets for each result
    for r in results:
        r["snippet"] = generate_snippet(r.pop("text"), params.q)

    return {
        "query": params.q,
        "k": params.k,
        "took_ms": round((time.time() - start) * 1000, 2),
        "results": results
    }

@router.get("/search/context")
def search_as_context(
    params: SearchRequest = Depends(),
    engine = Depends(get_engine)
):
    start = time.time()
    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    context = engine.search_as_context(
        params.q, top_k=params.k, max_tokens=params.max_tokens
    )
    return {
        "query": params.q,
        "context": context,
        "took_ms": round((time.time() - start) * 1000, 2),
    }


@router.get("/search/dual")
def search_dual(
    params: SearchRequest = Depends(),
    engine = Depends(get_engine)
):
    start = time.time()
    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if params.profile:
        results, profile_data = engine.search(params.q, params.k, use_dual=True, profile=True)
    else:
        results = engine.search(params.q, params.k, use_dual=True)
        profile_data = None

    response_results = []
    for doc_id, score in results:
        doc = engine.doc_store.get(doc_id)
        if not doc:
            continue
        snippet = generate_snippet(doc["text"], params.q)
        response_results.append(
            SearchResult(
                doc_id=doc_id,
                title=doc["title"],
                url=doc["url"],
                snippet=snippet,
                score=score,
            )
        )

    return SearchResponse(
        query=params.q,
        k=params.k,
        took_ms=round((time.time() - start) * 1000, 2),
        results=response_results,
        profile_data=profile_data
    )


@router.get("/search/fts")
def search_fts(
    params: SearchRequest = Depends(),
    engine = Depends(get_engine)
):
    start = time.time()
    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = engine.search(params.q, params.k, use_fts=True)

    response_results = []
    for doc_id, score in results:
        doc = engine.doc_store.get(doc_id)
        if not doc:
            continue
        snippet = generate_snippet(doc["text"], params.q)
        response_results.append(
            SearchResult(
                doc_id=doc_id,
                title=doc["title"],
                url=doc["url"],
                snippet=snippet,
                score=score,
            )
        )

    return SearchResponse(
        query=params.q,
        k=params.k,
        took_ms=round((time.time() - start) * 1000, 2),
        results=response_results,
    )

@router.get("/search/distributed", response_model=SearchResponse)
async def search_distributed(
    params: SearchRequest = Depends(),
    workers: str = Query(..., description="Comma-separated list of worker URLs (e.g., http://node1:8000,http://node2:8000)")
):
    start = time.time()

    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    worker_urls = [w.strip() for w in workers.split(",") if w.strip()]
    if not worker_urls:
        raise HTTPException(status_code=400, detail="No valid worker URLs provided")

    search_kwargs = {
        "agent_id": params.agent_id,
        "column_filter": params.column_filter,
        "use_fts": params.use_fts,
        "use_dual": params.use_dual,
        "profile": params.profile
    }

    results = await scatter_gather_search(
        params.q, params.k, worker_urls, **search_kwargs
    )
    
    # Note: When using scatter-gather, profiling metrics are node-specific and 
    # not easily aggregated at the coordinator level, so we omit profile_data here.

    return SearchResponse(
        query=params.q,
        k=params.k,
        took_ms=round((time.time() - start) * 1000, 2),
        results=results,
        profile_data=None
    )
