


# Crypto-RAG-bot

A Retrieval-Augmented Generation (RAG) bot tailored for cryptocurrency data and analysis. This bot ingests scraped crypto data, stores embeddings, performs retrieval, and generates human-readable answers using an LLM.

---

## 🚀 Features

- Ingest and index JSON / scraped crypto data into a vector store  
- Query the data using a retrieval + generation pipeline  
- Uses GenAI (Gemini) for text generation  
- Uses Pinecone as the vector database  
- Modular architecture: ingestion, embedding, retrieval, generation, API routes  

---

## 📦 Repository Structure

```

.
├── api.py             # Main API endpoints (e.g. REST handlers)
├── app.py             # Entry point / Web server initialization
├── query.py           # Query orchestration (retrieve + generate)
├── upload_json.py     # Script to upload JSON / scraped data to the index
├── requirements.txt   # Python dependencies
├── config/             # Configuration files (e.g. ENV templates, config settings)
├── embeddings/         # embedding model logic or wrapper
├── generation/         # generation / LLM logic (Gemini wrapper etc.)
├── retrieval/          # retrieval / vector search logic
├── routes/             # API route handlers if separated
├── utils/              # Utilities (logging, helpers, etc.)
├── vectorstore/        # Vector store / Pinecone interface logic
└── tests/              # Test suite

````

---

## 🛠️ Setup & Usage

### 1. Clone the repo

```bash
git clone https://github.com/irtaza129/Crypto-RAG-bot.git
cd Crypto-RAG-bot
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys & environment variables

Create a `.env` file in the project root (you can base it off `config/.env.template` if provided) with entries like:

```
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_index_name
DATA_FOLDER=path/to/your/scraped/json
```

Then in your code, load environment variables (e.g. via `python-dotenv` or `os.environ`) and avoid hardcoding keys.

### 4. Upload / index data

Run:

```bash
python upload_json.py
```

This script reads JSON files from `DATA_FOLDER`, computes embeddings, and indexes them into Pinecone under `INDEX_NAME`.

### 5. Run the application

```bash
python app.py
```

This will start the API or server (e.g. Flask / FastAPI) exposing endpoints defined in `api.py`.

### 6. Query the bot

You can send HTTP requests to the bot (via Postman, curl, or frontend) to query crypto topics. The `query.py` module will:

* retrieve top documents via vector search
* feed retrieved documents + user prompt into the Gemini LLM to generate an answer

### Streaming responses

This project exposes an SSE streaming endpoint so clients can receive answers progressively.

- Endpoint: `POST /query/stream`
- Content-Type: `application/json`
- Body: `{ "query": "Your question here" }`

Example using `curl` (keeps connection open and prints events as they arrive):

```bash
curl -N -X POST "http://localhost:8000/query/stream" \
	-H "Content-Type: application/json" \
	-d '{"query":"What are the FCA rules for crypto exchanges?"}'
```

Notes:
- The stream is Server-Sent Events (SSE) with `data:` messages containing
	chunks of the textual answer.
- After the text stream finishes the server sends an `event: metadata` message
	whose `data:` payload contains the `retrieved_chunks` JSON array.
- To implement true LLM-level streaming (avoid waiting for the full answer
	before emitting), integrate your LLM client in streaming/callback mode
	(Gemini/Google GenAI streaming) and yield parts as they're produced.

---

## 🧪 Testing

Run tests via:

```bash
pytest
```

Ensure your test environment has mock or dummy API keys and a test Pinecone instance or mocking in place.

---

## ✅ Best Practices & Tips

* **Never commit real API keys** into the repo — use `.env` + `.gitignore`
* To remove previously committed keys, consider rewriting git history (e.g. `git filter-branch` or BFG)
* Use GitHub Secrets if deploying via GitHub Actions or CI
* Monitor usage and quotas for Gemini & Pinecone
* Add error handling around API calls, rate limits, missing docs, etc.

---

## 👥 Contributing

Contributions are welcome! Feel free to open issues, suggest features, or submit pull requests. Please follow:

1. Fork repository
2. Create a feature branch
3. Write tests for your changes
4. Ensure linting / formatting
5. Submit PR describing your changes

---

## 📄 License

Specify your license here (MIT, Apache 2.0, etc.)

---
