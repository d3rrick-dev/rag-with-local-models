import os
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
        
        print(f"[Ollama Embedding] Processing vector calculations for {len(input)} text chunks...")
        
        with httpx.Client() as client:
            for index, text_chunk in enumerate(input):
                payload = {
                    "model": self.model_name,
                    "prompt": text_chunk
                }
                
                try:
                    response = client.post(self.url, json=payload, timeout=30.0)
                    
                    if response.status_code != 200:
                        print(f"[Ollama Embedding] ERROR: Server returned status {response.status_code} for chunk index {index}")
                        raise Exception(f"Ollama Embeddings engine returned error status: {response.status_code}")
                    
                    response_data = response.json()
                    vector = response_data.get("embedding")
                    
                    if not vector:
                        print(f"[Ollama Embedding] ERROR: No 'embedding' key found in response for chunk index {index}")
                        raise ValueError("Ollama response missing embedding array data.")
                    
                    # Convert values explicitly to floats to satisfy ChromaDB typing constraints
                    float_vector = [float(val) for val in vector]
                    embeddings_list.append(float_vector)
                    
                except Exception as e:
                    print(f"[Ollama Embedding] Connection or handling exception on chunk index {index}: {str(e)}")
                    raise e
                    
        print(f"[Ollama Embedding] Successfully generated {len(embeddings_list)} vector arrays.")
        return embeddings_list


class VectorRAGService:
    def __init__(self):
        # Configure ChromaDB with local persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Switch our embedding engine wrapper over to Ollama
        self.embedding_function = NativeOllamaEmbeddingFunction()
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="internal_knowledge_base",
            embedding_function=self.embedding_function
        )

    async def extract_text(self, upload_file: UploadFile) -> str:
        """
        Extracts raw textual content from uploaded TXT or PDF files.
        """
        filename = upload_file.filename.lower()
        
        # Reset the stream cursor back to the beginning of the file
        await upload_file.seek(0)
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
        """
        Splits long document text blocks into uniform chunks with overlapping windows.
        """
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
        """
        Generates unique IDs, binds chunks, and registers vectors into ChromaDB storage.
        Strips whitespace variations to keep parsing streams clean.
        """
        # Clean chunks to strip empty text blocks or raw carriage spaces
        cleaned_chunks = [chunk.strip() for chunk in text_chunks if chunk and chunk.strip()]
        
        if not cleaned_chunks:
            print("[ChromaDB] Warning: No non-empty text chunks found to store.")
            return

        ids = [f"{file_id}_chunk_{i}" for i in range(len(cleaned_chunks))]
        metadatas = [{"source_file": file_id} for _ in cleaned_chunks]
        
        # Batch insertions into ChromaDB step-by-step
        batch_size = 10
        for i in range(0, len(cleaned_chunks), batch_size):
            batch_end = i + batch_size
            
            self.collection.add(
                documents=cleaned_chunks[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            
        print(f"[ChromaDB] Successfully saved {len(cleaned_chunks)} clean chunks to collection storage.")
    def query_similar_context(self, query: str, max_results: int = 4) -> List[str]:
        """
        Queries ChromaDB collection to retrieve the most semantically relevant text chunks.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=max_results
        )
        return results["documents"][0] if results["documents"] else []