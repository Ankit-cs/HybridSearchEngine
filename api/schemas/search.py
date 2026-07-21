from pydantic import BaseModel, Field
from typing import Optional

class SearchRequest(BaseModel):
    q: str = Field(..., description="Search Query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    agent_id: Optional[str] = Field(None, description="Agent ID for memory context")
    column_filter: Optional[dict] = Field(None, description="Column filter: {column, value, op}")
    use_fts: bool = Field(False, description="Use persistent FTS index")
    use_dual: bool = Field(False, description="Use dual embeddings (content + context)")
    max_tokens: int = Field(4000, description="Max tokens for context assembly")

class SearchResult(BaseModel):
    doc_id: int
    title: str
    url: str
    score: float
    snippet: str

class SearchResponse(BaseModel):
    query: str
    k: int
    took_ms: float
    results: list[SearchResult]
