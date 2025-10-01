

# Document Ingestion API

> A full-stack solution for uploading, chunking, embedding, and storing documents (PDF/TXT) using FastAPI, MongoDB, and Pinecone vector storage. Supports chunking strategies and OpenAI embeddings for downstream retrieval and search tasks.

---


## Features

- **Upload PDF/TXT files** via a web UI or REST API
- **Chunk documents** using recursive or fixed strategies
- **Generate embeddings** with OpenAI
- **Store documents and chunks** in MongoDB
- **Optional Pinecone integration** for vector search
- **List all uploaded documents**
- **Retrieve all chunks for a document**
- **Delete documents and their chunks (MongoDB & Pinecone)**
- **Robust error handling and validation**
- **REST API** for programmatic access

## Architecture

- **Backend:** FastAPI ([main.py](main.py))
- **Database:** MongoDB
- **Vector DB:** Pinecone (optional)

---

## Getting Started

### Prerequisites

- Python 3.10+
- MongoDB instance (local or remote)
- OpenAI API key
- Pinecone account and API key (optional)

### Installation

1. **Clone the repository:**
	```sh
	git clone https://github.com/shishiradk/Document-Ingestion-API.git
	cd Document-Ingestion-API
	```
2. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```
3. **Create a `.env` file:**
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


---


## API Usage

### Upload Document

**POST** `/upload/`

- **Form fields:**
	- `file`: PDF or TXT file
	- `chunk_strategy`: `recursive` or `fixed`
- **Response:**
	- `doc_id`: Document ID in MongoDB
	- `num_chunks`: Number of chunks created
	- `vector_db_uploaded`: Whether Pinecone was used
	- `vector_db`: Which vector DB was used

**Example:**
```sh
curl -X POST "http://localhost:8000/upload/" \
	-F "file=@path/to/your/file.pdf" \
	-F "chunk_strategy=recursive"
```

---

### List Documents

**GET** `/documents/`

Returns a list of all uploaded documents with metadata.

**Example:**
```sh
curl http://localhost:8000/documents/
```

---

### Get Document Chunks

**GET** `/documents/{doc_id}/chunks`

Returns all chunks for a specific document.

**Example:**
```sh
curl http://localhost:8000/documents/<doc_id>/chunks
```

---

### Delete Document

**DELETE** `/documents/{doc_id}`

Deletes a document and its chunks from MongoDB and Pinecone (if enabled).

**Example:**
```sh
curl -X DELETE http://localhost:8000/documents/<doc_id>
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## Contact

Maintained by [shishiradk](https://github.com/shishiradk).
