from fastapi import APIRouter, Depends, HTTPException
import time
from api.deps import get_engine
from src.agent.llm_client import LLMClient
from api.schemas.search import SearchRequest

router = APIRouter(prefix="/api/v1", tags=["generate"])

@router.get("/search/generate")
def generate_answer(
    params: SearchRequest = Depends(),
    engine = Depends(get_engine)
):
    start = time.time()
    if not params.q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get context chunks (top 5 is usually good for LLMs to avoid distraction)
    context = engine.search_as_context(
        params.q, top_k=5, max_tokens=3000
    )
    
    llm = LLMClient()
    system_prompt = (
        "You are a highly intelligent AI search assistant. Your job is to read the provided context chunks "
        "and synthesize a direct, comprehensive answer to the user's query. "
        "IMPORTANT: You must use inline citations in the format [1], [2] to reference the document IDs provided in the context. "
        "If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n\n"
        "You MUST respond ONLY with a valid JSON object in the following format. Do not include markdown fences:\n"
        "{\n"
        '  "answer": "Your detailed answer with inline citations [1].",\n'
        '  "citations": [{"id": "1", "source": "Document Title", "text": "Relevant snippet..."}]\n'
        "}"
    )
    user_prompt = f"Context Documents:\n{context}\n\nUser Query: {params.q}\n\nGenerate Answer:"
    
    raw_response = llm.generate(system_prompt, user_prompt)
    
    import json
    import re
    try:
        clean_json = re.sub(r"```(?:json)?\s*", "", raw_response).strip("`").strip()
        parsed_answer = json.loads(clean_json)
    except Exception:
        parsed_answer = {
            "answer": raw_response,
            "citations": []
        }
    
    return {
        "query": params.q,
        "answer": parsed_answer.get("answer", raw_response),
        "citations": parsed_answer.get("citations", []),
        "took_ms": round((time.time() - start) * 1000, 2)
    }
