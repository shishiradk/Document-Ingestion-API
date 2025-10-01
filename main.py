from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from enum import Enum
from typing import Optional
import os, uuid
from datetime import datetime

# Load .env
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing!")

MONGO_URI = os.getenv("MONGO_URI")
USE_PINECONE = os.getenv("USE_PINECONE", "false").strip().lower() == "true"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") if USE_PINECONE else None
PINECONE_ENV = os.getenv("PINECONE_ENV") if USE_PINECONE else None
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "documents") if USE_PINECONE else None

# Chunking strategy enum
class ChunkStrategy(str, Enum):
    recursive = "recursive"
    fixed = "fixed"

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

# OpenAI Embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Pinecone setup
pinecone_index = None
if USE_PINECONE:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=1536,
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV or "us-east-1")
            )
        pinecone_index = pc.Index(PINECONE_INDEX)
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
        USE_PINECONE = False

# Utility functions
def extract_text(file: UploadFile) -> str:
    """Extract text from PDF or TXT files"""
    try:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            if len(reader.pages) == 0:
                raise HTTPException(status_code=400, detail="PDF file is empty")
            text = "".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
            return text
        elif file.filename.endswith(".txt"):
            content = file.file.read().decode("utf-8")
            if not content.strip():
                raise HTTPException(status_code=400, detail="TXT file is empty")
            return content
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Unable to decode file. Please ensure it's a valid text file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text: {str(e)}")

def chunk_text(text: str, strategy: ChunkStrategy) -> list[str]:
    """Chunk text using selected strategy"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Cannot chunk empty text")
    
    if strategy == ChunkStrategy.recursive:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)
    elif strategy == ChunkStrategy.fixed:
        # Fixed size chunking with no overlap
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid chunking strategy: {strategy}")
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks generated from text")
    
    return chunks

# API routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Document Ingestion API",
        "vector_db": "Pinecone" if USE_PINECONE else "MongoDB only"
    }

@app.post("/upload/")
async def upload_document(
    file: UploadFile, 
    chunk_strategy: ChunkStrategy = Form(ChunkStrategy.recursive)
):
    """
    Upload and process a document
    
    - **file**: PDF or TXT file to upload
    - **chunk_strategy**: Choose 'recursive' or 'fixed' chunking (default: recursive)
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        # Extract text
        text = extract_text(file)
        
        # Save full document in MongoDB with timestamp
        doc_id = documents_collection.insert_one({
            "filename": file.filename,
            "content": text,
            "chunk_strategy": chunk_strategy.value,
            "file_size": len(text),
            "timestamp": datetime.utcnow()
        }).inserted_id

        # Chunk text
        chunks = chunk_text(text, strategy=chunk_strategy)

        # Generate embeddings
        try:
            embedding_vectors = embeddings_model.embed_documents(chunks)
        except Exception as e:
            # Cleanup: remove document if embedding fails
            documents_collection.delete_one({"_id": doc_id})
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

        # Insert chunks into MongoDB (WITHOUT embeddings to save space)
        chunk_docs = [
            {
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": str(doc_id),
                "chunk_text": chunk,
                "chunk_index": i,
                "chunk_length": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
        chunks_collection.insert_many(chunk_docs)

        # Upload to Pinecone with proper metadata
        pinecone_uploaded = False
        if USE_PINECONE and pinecone_index:
            try:
                vectors_to_upsert = [
                    {
                        "id": f"{doc_id}_{i}",
                        "values": vector,
                        "metadata": {
                            "doc_id": str(doc_id),
                            "filename": file.filename,
                            "chunk_index": i,
                            "chunk_text": chunk[:500],  # First 500 chars (Pinecone metadata limit)
                            "chunk_strategy": chunk_strategy.value
                        }
                    }
                    for i, (chunk, vector) in enumerate(zip(chunks, embedding_vectors))
                ]
                
                # Upsert in batches of 100 (Pinecone limit)
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    pinecone_index.upsert(vectors=batch)
                
                pinecone_uploaded = True
            except Exception as e:
                print(f"Pinecone upload failed: {e}")
                # Don't fail the entire request if Pinecone fails

        return JSONResponse({
            "status": "success",
            "doc_id": str(doc_id),
            "filename": file.filename,
            "num_chunks": len(chunks),
            "chunk_strategy": chunk_strategy.value,
            "text_length": len(text),
            "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks),
            "vector_db_uploaded": pinecone_uploaded,
            "vector_db": "Pinecone" if USE_PINECONE else "None"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents"""
    try:
        docs = list(documents_collection.find({}, {
            "filename": 1, 
            "chunk_strategy": 1, 
            "file_size": 1,
            "timestamp": 1,
            "_id": 1
        }).limit(100))
        
        return {
            "total": len(docs),
            "documents": [
                {
                    "doc_id": str(doc["_id"]),
                    "filename": doc.get("filename"),
                    "chunk_strategy": doc.get("chunk_strategy"),
                    "file_size": doc.get("file_size"),
                    "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None
                }
                for doc in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

@app.get("/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    """Get all chunks for a specific document"""
    try:
        chunks = list(chunks_collection.find(
            {"doc_id": doc_id},
            {"_id": 0, "chunk_text": 1, "chunk_index": 1, "chunk_length": 1}
        ).sort("chunk_index", 1))
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found or has no chunks")
        
        return {
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chunks: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks from MongoDB and Pinecone"""
    try:
        # Validate ObjectId format
        try:
            obj_id = ObjectId(doc_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid document ID format")
        
        # Check if document exists
        doc = documents_collection.find_one({"_id": obj_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get all chunks before deleting (needed for Pinecone IDs)
        chunks = list(chunks_collection.find({"doc_id": doc_id}))
        
        # Delete from Pinecone first (if enabled)
        pinecone_deleted = False
        if USE_PINECONE and pinecone_index and chunks:
            try:
                ids_to_delete = [f"{doc_id}_{chunk['chunk_index']}" for chunk in chunks]
                if ids_to_delete:
                    pinecone_index.delete(ids=ids_to_delete)
                    pinecone_deleted = True
            except Exception as e:
                print(f"Pinecone deletion failed: {e}")
                # Continue with MongoDB deletion even if Pinecone fails
        
        # Delete from MongoDB
        documents_collection.delete_one({"_id": obj_id})
        chunks_deleted = chunks_collection.delete_many({"doc_id": doc_id}).deleted_count
        
        return {
            "status": "deleted",
            "doc_id": doc_id,
            "filename": doc.get("filename"),
            "chunks_deleted": chunks_deleted,
            "pinecone_deleted": pinecone_deleted
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

