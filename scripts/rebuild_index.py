"""Deletes all vectors from the configured Pinecone index, then uploads all JSON files
found in `Scrapped Data` (no skips). Use with caution: this fully clears the index.

Usage (Windows cmd):
    cd C:/Users/user/Documents/Crypto-RAG-bot
    python scripts/rebuild_index.py

The script will:
 - Delete all vectors from the index (tries multiple delete methods to support different Pinecone SDKs)
 - Iterate every .json file in `Scrapped Data` and upsert chunks in batches
 - Print a brief summary at the end

Make sure `.env` contains `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `GEMINI_API_KEY`.
"""
import os
import json
import sys
import pathlib
# ensure repo root is on sys.path so relative imports work when running scripts/*
repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from utils.logger import logger

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_FOLDER = os.getenv("DATA_FOLDER", "Scrapped Data")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not GEMINI_API_KEY:
    logger.error("Missing required env vars: PINECONE_API_KEY, PINECONE_INDEX_NAME, GEMINI_API_KEY")
    sys.exit(1)

# Init clients
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Helpers
def embed_text(text):
    result = genai.embed_content(model="models/embedding-001", content=text)
    return result["embedding"]

def chunk_text(text, chunk_size=2000, overlap=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Delete all vectors
logger.info("Deleting all vectors from Pinecone index...")
deleted = False
try:
    # Preferred: delete with delete_all flag
    try:
        index.delete(delete_all=True)
        logger.info("index.delete(delete_all=True) succeeded")
        deleted = True
    except TypeError:
        # Some clients expect no args or different method name
        index.delete_all()
        logger.info("index.delete_all() succeeded")
        deleted = True
except Exception as e:
    logger.warning(f"Could not delete using delete_all methods: {e}")
    # Try alternative: describe stats and delete by filter if available
    try:
        stats = index.describe_index_stats()
        logger.info(f"Index stats before delete attempt: {stats}")
    except Exception:
        logger.warning("Could not get index stats")

if not deleted:
    logger.warning("Proceeding to upload after attempting delete; index may still contain previous vectors.")

# Upload all JSON files in Scrapped Data
BATCH_SIZE = 100
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
logger.info(f"Found {len(files)} JSON files in {DATA_FOLDER}. Will upload all files.")

total_upserted = 0
for file_name in files:
    file_path = os.path.join(DATA_FOLDER, file_name)
    logger.info(f"Processing file: {file_name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # prefer common keys
            if 'data' in data and isinstance(data['data'], list):
                entries = data['data']
            elif 'items' in data and isinstance(data['items'], list):
                entries = data['items']
            else:
                # treat top-level dict as single entry
                entries = [data]
        else:
            entries = [ { 'text': str(data) } ]

        vectors_batch = []
        for idx, entry in enumerate(entries):
            # extract text
            if isinstance(entry, dict):
                text = entry.get('content') or entry.get('text') or entry.get('body') or str(entry)
                country_name = entry.get('country_name') or entry.get('country')
                source_name = entry.get('source_name') or entry.get('source') or file_name
                source_link = entry.get('source_link') or entry.get('url') or ''
            else:
                text = str(entry)
                country_name = None
                source_name = file_name
                source_link = ''

            if not text or not str(text).strip():
                continue

            jurisdiction = country_name or file_name.replace('.json','')

            for chunk_idx, chunk in enumerate(chunk_text(str(text))):
                try:
                    emb = embed_text(chunk)
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk {chunk_idx} of entry {idx} in {file_name}: {e}")
                    continue
                vid = f"{file_name.replace('.json','')}_{idx}_{chunk_idx}"
                if not source_link:
                    source_link = f"https://compliance-docs.example.com/{(jurisdiction or 'general').lower()}/{file_name}"
                metadata = {
                    'source': source_name,
                    'source_name': source_name,
                    'source_link': source_link,
                    'text': chunk[:300],
                    'jurisdiction': jurisdiction
                }
                if country_name:
                    metadata['country_name'] = country_name

                vectors_batch.append({'id': vid, 'values': emb, 'metadata': metadata})

            # Upsert in batches to avoid memory blowup
            if len(vectors_batch) >= BATCH_SIZE:
                index.upsert(vectors=vectors_batch)
                total_upserted += len(vectors_batch)
                logger.info(f"Upserted {len(vectors_batch)} vectors from {file_name} (batch)")
                vectors_batch = []

        # flush remaining
        if vectors_batch:
            index.upsert(vectors=vectors_batch)
            total_upserted += len(vectors_batch)
            logger.info(f"Upserted {len(vectors_batch)} vectors from {file_name} (final batch)")

    except Exception as e:
        logger.error(f"Failed processing {file_name}: {e}")

logger.info(f"Rebuild complete. Total vectors upserted (approx): {total_upserted}")
try:
    stats = index.describe_index_stats()
    logger.info(f"Index stats after rebuild: {stats}")
except Exception as e:
    logger.warning(f"Could not retrieve index stats after rebuild: {e}")

# update todo list: mark steps completed
print("DONE")
