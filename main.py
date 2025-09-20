from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import pinecone, os, uuid

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "documents")

# FastAPI setup
app = FastAPI(title="Document Ingestion API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["document_db"]
documents_collection = mongo_db["documents"]
chunks_collection = mongo_db["chunks"]

# LangChain Embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Optional Pinecone setup
if USE_PINECONE:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX, dimension=1536)
    pinecone_index = pinecone.Index(PINECONE_INDEX)

# Utility functions
def extract_text(file: UploadFile) -> str:
    """Extract text from PDF or TXT"""
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
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")

def chunk_text(text: str, strategy: str = "recursive") -> list[str]:
    """Chunk text using selected strategy"""
    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)
    elif strategy == "fixed":
        return [text[i:i+1000] for i in range(0, len(text), 1000)]
    else:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")

# API route
@app.post("/upload/")
async def upload_document(file: UploadFile, chunk_strategy: str = Form("recursive")):
    try:
        # Extract text
        text = extract_text(file)

        # Save full document in MongoDB
        doc_id = documents_collection.insert_one({
            "filename": file.filename,
            "content": text
        }).inserted_id

        # Chunk text
        chunks = chunk_text(text, strategy=chunk_strategy)

        # Generate embeddings in batch
        embedding_vectors = embeddings_model.embed_documents(chunks)

        # Insert chunks and embeddings into MongoDB
        chunks_collection.insert_many([
            {"doc_id": str(doc_id), "chunk": chunk, "embedding": vector}
            for chunk, vector in zip(chunks, embedding_vectors)
        ])

        # Optional Pinecone upload
        if USE_PINECONE:
            pinecone_index.upsert([
                (str(uuid.uuid4()), vector) for vector in embedding_vectors
            ])

        return JSONResponse({
            "status": "success",
            "doc_id": str(doc_id),
            "num_chunks": len(chunks),
            "pinecone": USE_PINECONE
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
