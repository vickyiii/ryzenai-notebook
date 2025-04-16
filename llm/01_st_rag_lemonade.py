# src/st_rag_lemonade.py
import streamlit as st
import time
import os
import warnings
import requests
import tiktoken

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community import document_loaders, vectorstores
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.docstore.document import Document

import logging
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

LEMONADE_BASE_URL = "http://localhost:8000/api/v0"
VECTOR_DB_DIR = "vector_dbs"

st.header("LLM RAG with Lemonade")

# Load available models from Lemonade
LEMONADE_MODEL_ID = "Llama-3.2-1B-Instruct-Hybrid"

# Input text to load the document
url_path = st.text_input("Enter the URL to load for RAG:",
                         value="https://www.gutenberg.org/cache/epub/75855/pg75855-images.html",
                         key="url_path")

# Select embedding type
embedding_type = st.selectbox("Please select an embedding type", ("huggingface", "fastembed"), index=0)

# Input for RAG
question = st.text_input("Enter the question for RAG:", value="What is this book about", key="question")

def count_tokens(text: str, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def load_document(url, max_tokens=2000):
    print("Loading document from URL...")
    st.markdown(''' :green[Loading document from URL...] ''')
    loader = document_loaders.WebBaseLoader(url)
    docs = loader.load()

    doc = docs[0]
    text = doc.page_content
    token_count = count_tokens(text)

    if token_count > max_tokens:
        st.warning(f"content too long({token_count} tokens), split into {max_tokens} tokens")
        words = text.split()
        while count_tokens(" ".join(words)) > max_tokens:
            words.pop()
        text = " ".join(words)
        doc = Document(page_content=text, metadata=doc.metadata)

    return [doc]

## Split the document into multiple chunks
def split_document(text, chunk_size=1000, overlap=100):
    print("Splitting document into chunks...")
    st.markdown(''' :green[Splitting document into chunks...] ''')
    text_splitter_instance = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter_instance.split_documents(text)

    
def initialize_embedding_fn(embedding_type="huggingface", model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"):
    print(f"Initializing {embedding_type} model with {model_name}...")
    st.write(f"Initializing {embedding_type} model with {model_name}...")

    if embedding_type == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)

    elif embedding_type == "fastembed":
        return FastEmbedEmbeddings(threads=16)

    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
## Create embeddings for these chunks of data and store it in chromaDB

def get_or_create_embeddings(document_url, embedding_fn, persist_dir=VECTOR_DB_DIR):
    vector_store_path = os.path.join(os.getcwd(), persist_dir)    
    start_time = time.time()
    print("No existing vector store found. Creating new one...")
    st.markdown(''' :green[No existing vector store found. Creating new one......] ''')
    document = load_document(document_url)
    documents = split_document(document)
    vector_store = vectorstores.Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=persist_dir
    )
    vector_store.persist()
    print(f"Embedding time: {time.time() - start_time:.2f} seconds")
    st.write(f"Embedding time: {time.time() - start_time:.2f} seconds")
    return vector_store
# Create the user prompt and generate the response
def handle_user_interaction(vector_store, chat_model):

    prompt_template ="""
    Use the following pieces of context to answer the question at the end as thoroughly as possible. 
    Be detailed, specific, and cover all relevant aspects mentioned in the context. 

    If the context seems incomplete, try to summarize based on what is available. 
    Do not answer with "I don't know" unless absolutely necessary.

    Context:
    {context}

    Question:
    {question}

    Answer in a detailed and informative manner:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    # Use retrievers to retrieve the data from the database
    st.markdown(''' :green[Using retrievers to retrieve the data from the database...] ''')
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    st.markdown(''' :green[Answering the query...] ''')
    qachain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever, chain_type="stuff", chain_type_kwargs=chain_type_kwargs)
    qachain.invoke({"query": "what is this book about?"})
    print(f"Model warmup complete...")
    st.markdown(''' :green[Model warmup complete...] ''')
       
    start_time = time.time()
    answer = qachain.invoke({"query": question})
    print(f"Answer: {answer['result']}")    
    print(f"Response time: {time.time() - start_time:.2f} seconds")
    st.write(f"Response time: {time.time() - start_time:.2f} seconds")
    
    return answer['result']

# Main Function to load the document, initialize the embeddings , create the vector database and invoke the model
def getfinalresponse(document_url, embedding_type, chat_model):    
    
    document_url = url_path    
    chat_model = LEMONADE_MODEL_ID      
    embedding_fn = initialize_embedding_fn(embedding_type)
    vector_store = get_or_create_embeddings(document_url, embedding_fn)     

    chat_model_instance = ChatOpenAI(
        base_url=LEMONADE_BASE_URL,
        model_name=chat_model,
        temperature=0.7,
        openai_api_key="none"
    )
    return handle_user_interaction(vector_store, chat_model_instance)

submit=st.button("Generate")

# generate response
if submit:    
    document_url = url_path    
    chat_model = LEMONADE_MODEL_ID
    
    with st.spinner("Loading document....🐎"):        
        st.write(getfinalresponse(document_url, embedding_type, chat_model))