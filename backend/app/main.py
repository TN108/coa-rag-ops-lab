from fastapi import FastAPI
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="RAG, Chain-of-Agents, hybrid retrieval, evaluation, and LLMOps lab.",
    version=settings.API_VERSION,
)


@app.get("/")
def root():
    return {
        "message": f"{settings.APP_NAME} API is running",
        "version": settings.API_VERSION,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.API_VERSION,
    }