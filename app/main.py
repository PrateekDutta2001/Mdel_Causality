"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes_select import router as select_router
from app.api.routes_status import router as status_router
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(
    title="Agentic RAG Causal Selector",
    description=(
        "Multi-agent LangGraph pipeline recommending pretrained models and "
        "hyperparameters via RAG retrieval and causal inference over experiment logs."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(status_router)
app.include_router(select_router)
