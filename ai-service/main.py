from fastapi import FastAPI
from api.summarize import router as summarize_router

app = FastAPI()

app.include_router(summarize_router)

@app.get("/")
def health():
    return {"statues": "ok"}