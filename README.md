# Document Ingestion API

Document Ingestion API is a full-stack solution for uploading, chunking, embedding, and storing documents (PDF/TXT) using FastAPI (backend), Streamlit (frontend), MongoDB, and optional Pinecone vector storage. It supports chunking strategies and OpenAI embeddings for downstream retrieval and search tasks.

## Features
- **Upload PDF/TXT files** via a simple web UI
- **Chunk documents** using recursive or fixed strategies
- **Generate embeddings** with OpenAI
- **Store documents and chunks** in MongoDB
- **Optional Pinecone integration** for vector search
- **REST API** for programmatic access

## Architecture
- **Backend:** FastAPI (see `main.py`)
- **Database:** MongoDB
- **Vector DB:** Pinecone

## Getting Started

### Prerequisites
- Python 3.10+
- MongoDB instance (local or remote)
- OpenAI API key
- Pinecone account and API key

### Installation
1. Clone the repository:
	```sh
	git clone https://github.com/shishiradk/Document-Ingestion-API.git
	cd Document-Ingestion-API
	```
2. Install dependencies:
	```sh
	pip install -r requirements.txt
	```
3. Create a `.env` file with the following variables:
	```env
	OPENAI_API_KEY=your_openai_api_key
	MONGO_URI=your_mongodb_uri
	USE_PINECONE=true|false
	PINECONE_API_KEY=your_pinecone_api_key
	PINECONE_ENV=your_pinecone_env
	PINECONE_INDEX=documents
	```
	Set `USE_PINECONE=false` to disable Pinecone integration.

### Running the Backend (FastAPI)
```sh
uvicorn main:app --reload
```

## API Usage

### Upload Document
`POST /upload/`
- **Form fields:**
  - `file`: PDF or TXT file
  - `chunk_strategy`: `recursive` or `fixed`
- **Response:**
  - `doc_id`: Document ID in MongoDB
  - `num_chunks`: Number of chunks created
  - `pinecone`: Whether Pinecone was used

## License
MIT License. See [LICENSE](LICENSE).
