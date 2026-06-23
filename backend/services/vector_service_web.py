from typing import List
from io import BytesIO
import httpx
from pypdf import PdfReader
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from fastapi import UploadFile

class NativeOllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.url = "http://localhost:11434/api/embeddings"
        self.model_name = "nomic-embed-text" 

    def __call__(self, input: Documents) -> Embeddings:
        embeddings_list = []
        
        with httpx.Client() as client:
            for text_chunk in input:
                payload = {
                    "model": self.model_name,
                    "prompt": text_chunk
                }
                
                response = client.post(self.url, json=payload, timeout=30.0)
                if response.status_code != 200:
                    raise Exception(f"Ollama Embeddings engine returned error status: {response.status_code}")
                
                response_data = response.json()
                embeddings_list.append(response_data["embedding"])
                
        return embeddings_list


class VectorRAGService:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        self.embedding_function = NativeOllamaEmbeddingFunction()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="internal_knowledge_base",
            embedding_function=self.embedding_function
        )

    async def extract_text(self, upload_file: UploadFile) -> str:
        filename = upload_file.filename.lower()
        content_bytes = await upload_file.read()
        
        if filename.endswith(".txt"):
            return content_bytes.decode("utf-8")
            
        elif filename.endswith(".pdf"):
            pdf_reader = PdfReader(BytesIO(content_bytes))
            extracted_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
            return extracted_text
            
        raise ValueError("Unsupported file format provided.")

    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
            if chunk_size <= overlap:
                break
        return chunks

    def store_document(self, file_id: str, text_chunks: List[str]):
        ids = [f"{file_id}_chunk_{i}" for i in range(len(text_chunks))]
        metadatas = [{"source_file": file_id} for _ in text_chunks]
        
        batch_size = 25
        for i in range(0, len(text_chunks), batch_size):
            batch_end = i + batch_size
            self.collection.add(
                documents=text_chunks[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

    def query_similar_context(self, query: str, max_results: int = 4) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=max_results
        )
        return results["documents"][0] if results["documents"] else []