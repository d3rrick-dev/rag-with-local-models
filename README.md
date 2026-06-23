Markdown
# Internal Corporate RAG Engine

A high-performance, asynchronous Retrieval-Augmented Generation (RAG) engine built with FastAPI and React. The system completely bypasses cloud API costs and limits by routing document embeddings and chat generation through a local **Ollama** runtime environment.

## Features

- **Local Vector Processing:** Generates text embeddings locally using `nomic-embed-text` via Ollama's modern `/api/embed` gateway.
- **Local Language Model:** Generates deterministic, strictly context-grounded completions using `gemma3:4b`.
- **Asynchronous Document Processing:** Handles multi-page text extraction from PDFs (`pypdf`) and text files with overlapping semantic sliding window chunking.
- **Persistent Vector Storage:** Leverages a local **ChromaDB** instance to cache and index document representations on disk.
- **Comprehensive Test Coverage:** Contains rigorous isolation testing pipelines using `pytest` and mock abstractions.

## Technical Stack

| Layer | Technology |
| :--- | :--- |
| **Backend Framework** | Python 3.11, FastAPI, Uvicorn |
| **Data Validation** | Pydantic v2, Pydantic Settings |
| **Vector Database** | ChromaDB (Local Persistent) |
| **Local AI Engine** | Ollama (`gemma3:4b`, `nomic-embed-text`) |
| **File Parsing** | PyPDF |
| **Testing** | Pytest, HTTPX TestClient |


## Code snippets
``` bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
ollama list

pip install -r requirements.txt

uvicorn backend.main:app --reload

#docs
http://127.0.0.1:8000/docs

#interactive site
http://localhost:8000/


#tests
pytest test_main.py -v
```


#site
![Sample Interraction](/output/sample.png)