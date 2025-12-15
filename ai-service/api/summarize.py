from fastapi import APIRouter, UploadFile, File 
import os

from ingestion.pdf_loader import extract_text
from ingestion.chunker import chunk_text
from services.summarizer import summarize_chunks

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR , exist_ok= True)

router = APIRouter()

@router.post("/summarize")
async def summarize_pdf(file : UploadFile = File(...)): 
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())
        
        text = extract_text(file_path)
        chunks = chunk_text(text)

        summary = summarize_chunks(chunks)

        return {
            "file_name": file.filename,
            "summary": summary
        }