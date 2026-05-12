from typing import Dict

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from .RagPipline import handle_uploaded_file, answer_question

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Aucun fichier n'a été fourni.")
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .pdf et .txt sont autorisés.")

    content = await file.read()
    result = handle_uploaded_file(file.filename, content)
    return {
        "message": "Document uploadé et indexé avec succès.",
        "details": result,
    }


@router.post("/question")
async def ask_question(payload: Dict[str, str] = Body(...)):
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Le champ 'question' est requis.")

    result = answer_question(question)
    return {
        "question": question,
        "answer": result,
    }
