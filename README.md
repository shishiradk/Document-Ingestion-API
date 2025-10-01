<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version">
</p>

# Document Ingestion API

> A full-stack solution for uploading, chunking, embedding, and storing documents (PDF/TXT) using FastAPI, MongoDB, and optional Pinecone vector storage. Supports chunking strategies and OpenAI embeddings for downstream retrieval and search tasks.

---

## Features

- **Upload PDF/TXT files** via a web UI or REST API
- **Chunk documents** using recursive or fixed strategies
- **Generate embeddings** with OpenAI
- **Store documents and chunks** in MongoDB
- **Optional Pinecone integration** for vector search
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
	- `pinecone`: Whether Pinecone was used

#### Example (using `curl`):

```sh
curl -X POST "http://localhost:8000/upload/" \
	-F "file=@path/to/your/file.pdf" \
	-F "chunk_strategy=recursive"
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
