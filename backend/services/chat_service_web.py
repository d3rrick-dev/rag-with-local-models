import os
import httpx
from backend.services.vector_service import VectorRAGService

class ChatGroundedService:
    def __init__(self, vector_service: VectorRAGService):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/responses"
        self.vector_service = vector_service

    async def execute_grounded_chat(self, user_query: str) -> str:
        """
        Retrieves relevant local vectors, shapes a strict system prompt constraint,
        and returns a grounded AI response using the native HTTP /v1/responses pipeline.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is missing from server configurations.")

        # Fetch vector documentation matches
        matching_chunks = self.vector_service.query_similar_context(user_query)
        context_block = "\n---\n".join(matching_chunks)

        # definitions
        system_prompt = (
            "You are a secure internal corporate AI assistant.\n"
            "Your answer MUST be based strictly and entirely on the internal document context provided below.\n"
            "If the answer cannot be confidently derived from the context block, you must state exactly: "
            "'I cannot find this information in the internal documents.'\n"
            "Do not use external knowledge or invent facts under any circumstance."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        formatted_input = f"SYSTEM RULES:\n{system_prompt}\n\nINTERNAL CONTEXT:\n{context_block}\n\nUSER QUESTION: {user_query}"

        payload = {
            "model": "gpt-5.4-mini",
            "input": formatted_input,
            "store": True
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.url, json=payload, headers=headers, timeout=15.0)
                
                if response.status_code == 401:
                    raise Exception("Invalid OpenAI API key provided.")
                elif response.status_code == 429:
                    raise Exception("OpenAI API rate limit exceeded. Please try again later.")
                elif response.status_code != 200:
                    raise Exception(f"OpenAI API returned an unexpected error status: {response.status_code}")
                
                response_data = response.json()
                
                try:
                    outputs = response_data.get("output", [])
                    if outputs:
                        contents = outputs[0].get("content", [])
                        if contents:
                            extracted_text = contents[0].get("text", "")
                            if extracted_text:
                                return extracted_text.strip()
                    
                    raise ValueError("OpenAI response structure was valid JSON but lacked text output.")
                    
                except (KeyError, IndexError, AttributeError) as parse_err:
                    raise Exception(f"Failed to parse OpenAI nested response structure: {str(parse_err)}")

            except httpx.TimeoutException:
                raise Exception("The request to OpenAI API timed out.")
            except httpx.RequestError as e:
                raise Exception(f"Network error while communicating with OpenAI: {str(e)}")