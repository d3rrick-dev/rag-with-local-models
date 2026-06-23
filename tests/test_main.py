from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch
from backend.main import app

client = TestClient(app)


def test_upload_document_forbidden_role():
    """
    Ensures that a non-admin role receives a 403 Forbidden response status.
    """
    file_payload = {"file": ("test.txt", b"Dummy text corporate data content.")}
    form_payload = {"role": "user"}

    response = client.post("/api/admin/upload", files=file_payload, data=form_payload)
    
    assert response.status_code == 403
    assert response.json()["detail"] == "Access Denied: Only platform administrators can upload assets."


@patch("main.vector_service")
def test_upload_document_empty_content(mock_vector_service):
    """
    Ensures that a file yielding empty text returns a 400 Bad Request status.
    """
    # Mocking extraction to return an empty string
    mock_vector_service.extract_text = AsyncMock(return_value="")
    
    file_payload = {"file": ("empty.txt", b"")}
    form_payload = {"role": "admin"}

    response = client.post("/api/admin/upload", files=file_payload, data=form_payload)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "The submitted file contains no readable textual data."


@patch("main.vector_service")
def test_upload_document_success(mock_vector_service):
    """
    Ensures a valid text file uploaded by an admin is successfully chunked and stored.
    """
    mock_vector_service.extract_text = AsyncMock(return_value="Valid corporate policy document.")
    mock_vector_service.chunk_text = MagicMock(return_value=["Valid corporate policy document."])
    mock_vector_service.store_document = MagicMock()

    file_payload = {"file": ("policy.txt", b"Valid corporate policy document.")}
    form_payload = {"role": "admin"}

    response = client.post("/api/admin/upload", files=file_payload, data=form_payload)
    
    assert response.status_code == 201
    assert response.json()["filename"] == "policy.txt"
    assert response.json()["chunks_processed"] == 1
    
    mock_vector_service.store_document.assert_called_once_with(
        file_id="policy.txt", 
        text_chunks=["Valid corporate policy document."]
    )


def test_chat_query_empty_payload():
    """
    Ensures that sending an empty string to the query endpoint returns a 400 Bad Request status.
    """
    payload = {"message": "   "}
    response = client.post("/api/chat/query", json=payload)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Message query string cannot be empty."


@patch("main.chat_service")
def test_chat_query_success(mock_chat_service):
    """
    Ensures that a valid user query successfully calls the execution service and returns the text reply.
    """
    mock_chat_service.execute_grounded_chat = AsyncMock(return_value="Grounded response from documents.")
    
    payload = {"message": "What is the corporate remote policy?"}
    response = client.post("/api/chat/query", json=payload)
    
    assert response.status_code == 200
    assert response.json()["reply"] == "Grounded response from documents."
    mock_chat_service.execute_grounded_chat.assert_called_once_with("What is the corporate remote policy?")


# Component Isolation Logic Unit Tests
def test_vector_service_chunking():
    """
    Verifies that the text chunking mechanism functions correctly independent of external dependencies.
    """
    from backend.services.vector_service import VectorRAGService
    
    # Patch out database instantiation during initialization
    with patch("chromadb.PersistentClient"):
        service = VectorRAGService()
        sample_text = "A" * 1000
        chunks = service.chunk_text(sample_text, chunk_size=600, overlap=100)
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 600