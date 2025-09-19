from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from io import BytesIO
import os, uuid, re
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#Index Name
index_name = "document-ingestion-api"

#initializing pinecone client connection
pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # depends on embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

def extract_text(file: UploadFile) -> str:
    """Extract text from a PDF or TXT file"""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    elif file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    else:
        raise HTTPException(status_code=400,detail="Only PDF and TXT file are supported ")

def chunk_text(text: str, strategy: str = "recursive") -> list[str]:
    """Splits test into chunks the seleced strategy."""
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    else:
        raise ValueError(" Invalid chunking strategy")
    return splitter.split_text(text)

# generating embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

