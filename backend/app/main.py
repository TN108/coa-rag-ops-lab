from fastapi import FastAPI
from app.config import settings
from app.api.v1 import upload
from app.api.v1.rag_router import router as rag_router
from app.api.v1.coa_router import router as coa_router
from app.api.v1.coa_eval_router import router as coa_eval_router
from app.api.v1.rag_eval_router import router as rag_eval_router

app = FastAPI(
    title=settings.APP_NAME,
    description="RAG, Chain-of-Agents, hybrid retrieval, evaluation, and LLMOps lab.",
    version=settings.API_VERSION,
)

app.include_router(upload.router)
app.include_router(rag_router, prefix="/api/v1")
app.include_router(coa_router, prefix="/api/v1")
app.include_router(coa_eval_router, prefix="/api/v1")

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
app.include_router(rag_eval_router, prefix="/api/v1")