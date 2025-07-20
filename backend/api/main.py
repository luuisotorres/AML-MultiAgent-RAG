from fastapi import FastAPI
from backend.api.routes import query

app = FastAPI(title="AML MultiAgent RAG API")

app.include_router(
    query.router,
    prefix="/api/v1",
)
