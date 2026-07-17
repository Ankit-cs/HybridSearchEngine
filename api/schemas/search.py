from pydantic import BaseModel, Field
from typing import List

class SearchRequest(BaseModel):
    q: str =  Field(..., description="Search Query")
    k: int = Field(10, ge=1, le=100, description="Number of results")

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
    
      