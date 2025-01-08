# Rag Agent Service

This project provides a secure Retrieval-Augmented Generation (RAG) agent service using FastAPI. The service allows you to store documents in a vector space and query them using LLM.

## Features

- Store documents in a vector space (instead of in-memory vector store used session-based storage to save costs)
- Query the vector space using LLM
- Supports multiple document formats (PDF, etc.)

## Requirements

- Python 3.8+
- pip

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/abhishekshingadiya/rag-agent.git
    cd secure-rag-agent
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    ```sh
    cp env.example .env
    # Edit .env to add your OPENAI_API_KEY
    ```

## Usage

1. Start the FastAPI server:
    ```sh
    uvicorn app:app --host 0.0.0.0 --port 8001
    ```

2. Access the health check endpoint:
    ```sh
    curl http://localhost:8001/health
    ```

3. Store documents in the vector space:
    ```sh
    curl -X POST "http://localhost:8001/secure-rag-agent/train" -H "Content-Type: application/json" -d '{"document_paths":["./data/handbook.pdf"]}'
    ```

4. Query the vector space(add session_id to reuse the vector space):
    ```sh
    curl -X POST "http://localhost:8001/secure-rag-agent/query" -H "Content-Type: application/json" -d '{"questions": ["What is the name of the company?", "Who is PM of India?", "Who is the CEO of the company?", "What is their vacation policy?", "What is the termination policy?"], "session_id": "your_session_id"}'
    ```

5. Store documents and query the vector space:
    ```sh
    curl -X POST 'http://localhost:8001/secure-rag-agent' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"document_paths":["./data/handbook.pdf"], "questions":["What is the name of the company?", "Who is PM of India?", "Who is the CEO of the company?", "What is their vacation policy?", "What is the termination policy?"]}'
    ```

## Endpoints

- `GET /health`: Check if the service is running.
- `POST /secure-rag-agent`: Store documents and query the vector space.
- `POST /secure-rag-agent/train`: Store documents in the vector space.
- `POST /secure-rag-agent/query`: Query the vector space using LLM.

## License

This project is licensed under the MIT License.