from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
import json
import time

from src.query.search import SearchEngine
from src.utils.config import (
    INVERTED_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    METADATA_PATH
)

from api.routes.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # STARTUP
    start = time.time()

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    engine = SearchEngine(
        INVERTED_INDEX_PATH,
        DOCUMENT_STORE_PATH,
        total_docs=metadata["total_docs"]
    )

    app.state.engine = engine

    print(f"Engine loaded in {round(time.time() - start, 2)}s")



    yield  #app runs while paused here

    
    # SHUTDOWN   
    print("Shutting down AstraSearch API")


app = FastAPI(
    title="AstraSearch API",
    version="1.0",
    description="Search engine API powered by BM25 and title weighting",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)


@app.get("/health")
def health():
    return {"status": "ok"}
