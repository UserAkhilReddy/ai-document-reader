from fastapi import APIRouter
from pydantic import BaseModel

from ingestion.pdf_loader import extract_text
from ingestion.chunker import chunk_text
from services.vector_store import create_vector_store, search
from services.embeddings import embed_query
from services.qa import answer_question

router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask(req: AskRequest):
    pdf_path = "uploaded_files/AKHIL_JAVA_RESUME.pdf"

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    index, stored_chunks = create_vector_store(chunks)

    query_vec = embed_query(req.question)
    relevant_chunks = search(index, stored_chunks, query_vec, top_k=3)

    answer = answer_question(req.question, relevant_chunks)

    return {
        "question": req.question,
        "answer": answer
    }
