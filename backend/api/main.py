from fastapi import FastAPI
from backend.api.routes import query
from backend.api.routes.multi_agent import multi_agent_router

app = FastAPI(
    title="AML MultiAgent RAG API",
    description=(
        "Multi-Agent RAG system for AML compliance queries with "
        "both single-agent (fast) and multi-agent (validated) endpoints"
    ),
    version="1.0.0"
)

# Single RAG agent endpoint (fast, simple responses)
app.include_router(
    query.router,
    prefix="/api/v1",
)

# Multi-agent endpoints (enhanced quality validation)
app.include_router(
    multi_agent_router,
    prefix="/api/v1/multi-agent",
)
