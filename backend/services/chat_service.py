import os
import httpx
from backend.services.vector_service import VectorRAGService

class ChatGroundedService:
    def __init__(self, vector_service: VectorRAGService):
        self.url = "http://localhost:11434/api/generate"
        self.model_name = "gemma3:4b"
        self.vector_service = vector_service

    async def execute_grounded_chat(self, user_query: str) -> str:
        """
        Retrieves relevant local vectors, shapes a strict prompt constraint,
        and prompts Ollama via a native HTTP payload channel.
        """
        matching_chunks = self.vector_service.query_similar_context(user_query)
        context_block = "\n---\n".join(matching_chunks)

        system_prompt = (
            "You are a secure internal corporate AI assistant.\n"
            "Your answer MUST be based strictly and entirely on the internal document context provided below.\n"
            "If the answer cannot be confidently derived from the context block, you must state exactly: "
            "'I cannot find this information in the internal documents.'\n"
            "Do not use external knowledge or invent facts under any circumstance."
        )

        formatted_prompt = f"SYSTEM RULES:\n{system_prompt}\n\nINTERNAL CONTEXT:\n{context_block}\n\nUSER QUESTION: {user_query}"

        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.url, json=payload, timeout=60.0)
                
                if response.status_code != 200:
                    raise Exception(f"Ollama generation engine returned error status: {response.status_code}")
                
                response_data = response.json()
                extracted_text = response_data.get("response", "")
                
                if not extracted_text:
                    raise ValueError("Ollama completed execution but structural text output block was empty.")
                    
                return extracted_text.strip()

            except httpx.TimeoutException:
                raise Exception("The request to the local Ollama daemon timed out.")
            except httpx.RequestError as e:
                raise Exception(f"Network error while communicating with Ollama process: {str(e)}")