import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ConfyBot", layout="wide")
st.title("🤖 ConfyBot - Ask Me About the Conference!")

# Load PDF
loader = PyPDFLoader("Conference.pdf")
docs = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs_chunks = text_splitter.split_documents(docs)

# Vector store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs_chunks, embeddings)

# Q&A Chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=db.as_retriever())

# User input
query = st.text_input("Ask me anything about the conference:")

if query:
    response = qa.run(query)
    st.write("💬", response)
