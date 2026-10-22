from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from core.config import settings
from api.routes_files import router as files_router

app = FastAPI(
    title="RAG Application",
    description="RAG based document answering",
    version="1.0.0"
)

app.mount("/static",StaticFiles(directory="static"), name="static")

app.include_router(files_router, prefix="/api/v1", tags={"files"})

@app.get("/")
async def root():
    """Serving the HTML UI"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        PORT=settings.PORT,
        reload=settings.DEBUG
    )    