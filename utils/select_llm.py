import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

LLM_PROVIDER= os.getenv("LLM_PROVIDER", "openai")
DEFAULT_EMB_MODEL= os.getenv("DEFAULT_EMB_MODEL", "text-embedding-3-small")
DEFAULT_LLM_MODEL= os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")

def get_embedding_langchain(llm_provider=LLM_PROVIDER, model_name=DEFAULT_EMB_MODEL):
    if llm_provider == "openai":
        return OpenAIEmbeddings(model=model_name)


def get_llm_model_langchain(llm_provider=LLM_PROVIDER, model_name=DEFAULT_LLM_MODEL, **kwargs):
    if llm_provider == "openai":
        return ChatOpenAI(temperature=0.1, model_name=model_name, **kwargs)
