from fastapi import FastAPI

from api.summarize import router as summarize_router
from api.ask import router as ask_router

app = FastAPI(title="AI Document Reader")


app.include_router(summarize_router)
app.include_router(ask_router)


@app.get("/")
def health_check():
    return {"status": "AI Document Reader is running"}
