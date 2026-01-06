import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from utils.logger import logger

# =====================================================
# Load environment variables
# =====================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_FOLDER = os.getenv("DATA_FOLDER", "Scrapped Data")
UPLOADED_FILE_TRACKER = "uploaded_files.json"

if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables")

# =====================================================
# Load uploaded files tracker
# =====================================================
if os.path.exists(UPLOADED_FILE_TRACKER):
    with open(UPLOADED_FILE_TRACKER, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            uploaded_files = set(data.keys())
        elif isinstance(data, list):
            uploaded_files = set(data)
        else:
            uploaded_files = set()
else:
    uploaded_files = set()

logger.info(f"Skipping {len(uploaded_files)} already-uploaded files")

# =====================================================
# Pinecone init
# =====================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

# =====================================================
# Gemini
# =====================================================
genai.configure(api_key=GEMINI_API_KEY)

def embed_text(text: str):
    return genai.embed_content(
        model="models/embedding-001",
        content=text
    )["embedding"]

# =====================================================
# Chunking
# =====================================================
def chunk_text(text, chunk_size=2000, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# =====================================================
# Batching
# =====================================================
BATCH_SIZE = 50

def batch_iter(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]

# =====================================================
# Ingestion
# =====================================================
logger.info(f"Starting ingestion from: {DATA_FOLDER}")

uploaded_this_run = []

for file_name in os.listdir(DATA_FOLDER):
    if not file_name.endswith(".json"):
        continue

    if file_name in uploaded_files:
        logger.info(f"[SKIP] Already uploaded: {file_name}")
        continue

    file_path = os.path.join(DATA_FOLDER, file_name)
    logger.info(f"[PROCESS] {file_name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = data if isinstance(data, list) else [data]
        vectors = []

        for doc_idx, doc in enumerate(documents):
            if not isinstance(doc, dict):
                continue

            content = (doc.get("content") or "").strip()
            if not content:
                continue

            country_name = doc.get("country_name", "Global")
            metadata_base = {
                "title": doc.get("title"),
                "category": doc.get("category"),
                "date": doc.get("date"),
                "source_name": doc.get("source_name"),
                "source_link": doc.get("source_link"),
                "country_name": country_name,
                "jurisdiction": country_name
            }

            for chunk_idx, chunk in enumerate(chunk_text(content)):
                vectors.append({
                    "id": f"{file_name}_{doc_idx}_{chunk_idx}",
                    "values": embed_text(chunk),
                    "metadata": {
                        **metadata_base,
                        "chunk_text": chunk[:1200]
                    }
                })

        uploaded_count = 0
        for batch in batch_iter(vectors, BATCH_SIZE):
            index.upsert(vectors=batch)
            uploaded_count += len(batch)

        logger.info(f"[OK] Uploaded {uploaded_count} vectors from {file_name}")

        # Mark file as uploaded ONLY after success
        uploaded_files.add(file_name)
        uploaded_this_run.append(file_name)
        with open(UPLOADED_FILE_TRACKER, "w", encoding="utf-8") as f:
            json.dump(list(uploaded_files), f, indent=2)

    except Exception as e:
        logger.error(f"[ERROR] Failed {file_name}: {e}")

# =====================================================
# Summary of uploaded files
# =====================================================
if uploaded_this_run:
    logger.info("Upload complete. Files uploaded in this run:")
    for fn in uploaded_this_run:
        logger.info(f"  - {fn}")
else:
    logger.info("No new files uploaded in this run.")
