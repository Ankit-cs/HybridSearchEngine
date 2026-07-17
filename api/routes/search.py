from fastapi import APIRouter, Depends, HTTPException
import time
from src.utils.snippet import generate_snippet

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

    results = engine.search(params.q, params.k)

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
                    score=score
                )
            )

    return SearchResponse(
        query=params.q,
        k=params.k,
        took_ms=round((time.time() - start) * 1000, 2),
        results=response_results
    )
