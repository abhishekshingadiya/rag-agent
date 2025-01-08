import os
from datetime import datetime

from fastapi import (FastAPI, Body, Response)
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from services.chat import llm_response
from services.train import vector_space_creator, store_documents

app = FastAPI()

if not os.environ.get("OPENAI_API_KEY"):
  print("OPENAI_API_KEY not found, please add into your environment variables")

@app.get("/health")
def read_root():
    return "No worries, Your Secure Rag Agent Service is running Healthy! "

@app.post("/secure-rag-agent")
def secure_rag(response: Response, payload: dict = Body(...),
               # authorization: str = Header(None) # For secure use case
               ):
    """
    Function to perform secure rag Q&A
    :param response:
    :param payload:
    :return:
    """
    document_paths= payload.get("document_paths")
    rag_queries= payload.get("questions")
    session_id= payload.get("session_id", datetime.now().strftime("%Y%m%d%H%M%S"))

    # get vector store
    vector_store = vector_space_creator(session_id)
    # store documents
    if store_documents(vector_store,document_paths):
        # query the vector space
        response= llm_response(vector_store,rag_queries)
        return {"answers": response, "session_id": session_id}
    else:
        response.status_code= 500
        return {"message": f"Error while storing documents in the vector space {session_id}"}

@app.post("/secure-rag-agent/train")
def secure_rag_train(response: Response, payload: dict = Body(...),
                     # authorization: str = Header(None) # For secure use case
                     ):
    """
    Function to store documents in the vector space
    :param response:
    :param payload:
    :return:
    """
    document_paths= payload.get("document_paths")
    session_id= payload.get("session_id", datetime.now().strftime("%Y%m%d%H%M%S"))

    # get vector store
    vector_store = vector_space_creator(session_id)
    # store documents
    if store_documents(vector_store,document_paths):
        return {"message": f"Documents stored successfully in the vector space {session_id}","session_id": session_id}
    else:
        response.status_code= 500
        return {"message": f"Error while storing documents in the vector space {session_id}", "session_id": session_id}

@app.post("/secure-rag-agent/query")
def secure_rag_query(response: Response, payload: dict = Body(...),
                     # authorization: str = Header(None) # For secure use case
                     ):
    """
    Function to query the vector space
    :param response:
    :param payload:
    :return:
    """
    rag_queries= payload.get("questions")
    session_id= payload.get("session_id", datetime.now().strftime("%Y%m%d%H%M%S"))

    # get vector store
    vector_store= vector_space_creator(session_id)
    response= llm_response(vector_store,rag_queries)
    return response

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        log_level="debug",
        reload=False,
        loop="asyncio",
        port=8001
    )

