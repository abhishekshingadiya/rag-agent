import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from utils.prompts import rag_prompt_template
from services.train import rag_query_data
from utils.logging_helpers import logger
from utils.select_llm import get_llm_model_langchain

EVALUATION = os.getenv('EVALUATION', 'False').lower() in ('true', '1', 't', 'y', 'yes')


def openai_response(prompt):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],

    )
    return response.choices[0].message.content.strip()


def llm_response(vector_store, questions):
    """
    Function to query the vector space and get the response from LLM
    :param vector_store:
    :param questions:
    :return:
    """
    responses = []
    for question in questions:
        try:
            chunks, chunks_str = rag_query_data(vector_store, question)
            prompt = rag_prompt_template.format(chunks=chunks_str, question=question)
            response = openai_response(prompt)
            responses.append({"question": question, "response": response, "chunks": chunks})
            logger.info(f"Response for question: {question} is: {response} with chunks: {chunks}")
            if EVALUATION:
                # if evaluation is enabled, perform evaluation
                pass
        except Exception as e:
            logger.error(f"Error while querying the model for question: {question}: {str(e)}")
    return responses


def llm_response_langchain(vector_store, questions):
    """
    Function to query the vector space and get the response from LLM
    :param vector_store:
    :param questions:
    :return:
    """
    llm_model = get_llm_model_langchain()
    responses = []
    for question in questions:
        try:
            chunks, chunks_str = rag_query_data(vector_store, question)
            prompt = PromptTemplate(template=rag_prompt_template, input_variables=["chunks", "question"])
            parser = StrOutputParser()
            chain = prompt | llm_model | parser
            response = chain.invoke({"chunks": chunks, "question": question})
            responses.append({"question": question, "response": response, "chunks": chunks})
            logger.info(f"Response for question: {question} is: {response} with chunks: {chunks}")
            if EVALUATION:
                # if evaluation is enabled, perform evaluation
                pass
        except Exception as e:
            logger.error(f"Error while querying the model for question: {question}: {str(e)}")
    return responses