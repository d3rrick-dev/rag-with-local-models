import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.services.vector_service import VectorRAGService
from backend.services.chat_service import ChatGroundedService
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Internal Corporate RAG Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_service = VectorRAGService()
chat_service = ChatGroundedService(vector_service=vector_service)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/admin/upload", status_code=status.HTTP_201_CREATED)
async def upload_internal_document(
    file: UploadFile = File(...),
    role: str = Form(...)
):
    """
    Administrative gateway for target file parsing, vector ingestion, and embedding storage.
    Simulates Role-Based Access Control via form attribute assertion.
    """
    if role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Denied: Only platform administrators can upload assets."
        )

    try:
        extracted_text = await vector_service.extract_text(file)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The submitted file contains no readable textual data."
            )

        chunks = vector_service.chunk_text(extracted_text)
        
        vector_service.store_document(file_id=file.filename, text_chunks=chunks)

        return {
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "status": "Successfully ingested and vector-indexed."
        }

    except ValueError as format_exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(format_exception)
        )
    except Exception as server_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion lifecycle breakdown: {str(server_error)}"
        )


@app.post("/api/chat/query", response_model=ChatResponse)
async def execute_grounded_query(payload: ChatRequest):
    """
    User chat conversational endpoint running clean context grounding checks.
    """
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message query string cannot be empty."
        )

    try:
        ai_response = await chat_service.execute_grounded_chat(payload.message)
        return ChatResponse(reply=ai_response)
        
    except Exception as runtime_error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating response: {str(runtime_error)}"
        )

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")