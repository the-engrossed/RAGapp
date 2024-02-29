import json
import os
import sys
import boto3

#Using Titan Embeddings Model to generate Embeddings

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# For Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

#Vector Embedding and Vector Store

from langchain.vectorstores import FAISS

#LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#BedRock Client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embedding = BedrockEmbeddings(model_id ="amazon.titan-text-express-v1", client=bedrock)

#Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)
    
    docs= text_splitter.split_documents(documents)
    return docs

#Vector Embedding and Vector Store

def get_vectorstore(docs):
    vectorstore_faiss= FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    #create anthropic model
    llm = Bedrock(model_id ="ai21.j2-mid-v1",client=bedrock,
                  model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    #create anthropic model
    llm = Bedrock(model_id ="meta.llama2-70b-chat-v1",client=bedrock,
                  model_kwargs={'max_gen_len':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end and summarize with atleast 
250 words with detailed explanations. If you don't know the answer, 
just say I do not have an answer.
<context>
{context}
</context

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context",'question'])
