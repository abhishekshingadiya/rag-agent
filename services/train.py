import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logging_helpers import logger
from utils.select_llm import get_embedding_langchain


def process_file_reader(file_path):
    """
    Function to read the file and return the documents
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1][1:].lower()
    try:
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
        # for other file formats
        # elif ext == "txt" or ext == "md":
        #     loader = TextLoader(file_path)
        # elif ext == "csv":
        #     loader = CSVLoader(file_path)
        # elif ext == "xml":
        #     loader = UnstructuredXMLLoader(file_path)
        # elif ext == "xlsx":
        #     loader = UnstructuredExcelLoader(file_path)
        # elif ext == "json":
        #     loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        # elif ext == "docx":
        #     loader = Docx2txtLoader(file_path)
        # elif ext == "jsonl":
        #     loader = JSONLoader(
        #         file_path, jq_schema=".content", text_content=False, json_lines=True
        #     )
        else:
            logger.error(f"Unsupported file format received {ext} for {file_path}")
            return []
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error while reading file {file_path}: {str(e)}")
        return []


def split_text(documents):
    """
    Function to split the text into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    return chunks


def vector_space_creator(session_id):
    """
    Function to create a vector space
    :return:
    """
    vector_store = Chroma(
        collection_name=session_id,
        embedding_function=get_embedding_langchain(),
        persist_directory=f"./chroma_db/{session_id}",  # Where to save data locally, remove if not necessary
    )
    return vector_store


def store_documents(vector_store, document_paths):
    """
    Function to store the documents in the vector space
    :param vector_store:
    :param document_paths:
    :return:
    """
    try:
        documents = []
        for document_path in document_paths:
            document = process_file_reader(document_path)
            document = split_text(document)
            documents.extend(document)
            logger.info(f"{document_path} is processed")
        vector_store.add_documents(documents=documents)
        return True
    except Exception as e:
        logger.error(f"Error while storing documents: {str(e)}")
        return False


def rag_query_data(vector_store, query):
    """
    Function to query the vector space
    :param vector_store:
    :param query:
    :return:
    """
    chunks = vector_store.similarity_search_by_vector(
        embedding=get_embedding_langchain().embed_query(query), k=4
    )
    chunks_str = ""
    for i, chunk in enumerate(chunks):
        chunks_str += f"\n\n{chunk.page_content}"
    return chunks, chunks_str


# this function doing the same thing as rag_query_data but with different approach
def rag_query_by_retriever(vector_store, query):
    """
    Function to query the vector space by retriever
    :param vector_store:
    :param query:
    :return:
    """
    retriever = vector_store.as_retriever()
    results = retriever.invoke(query)
    return results
