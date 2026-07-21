from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
import json
import time

from src.query.search import SearchEngine
from src.utils.config import (
    INVERTED_INDEX_PATH,
    DOCUMENT_STORE_PATH,
    METADATA_PATH,
    CATALOG_DB_PATH,
)

from api.routes.search import router as search_router
from api.routes.agentic import router as agentic_router


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    print(f"Catalog: {engine.get_catalog_stats()}")

    yield

    print("Shutting down AstraSearch API")
    engine.catalog.close()


app = FastAPI(
    title="AstraSearch API",
    version="2.0",
    description="AI-Powered Hybrid Search Engine with ACID, Time-Travel, Dual Embeddings, and Agent Memory",
    lifespan=lifespan,
)

import os

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.include_router(agentic_router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}
