
import os
import json
from pinecone import Pinecone
import google.generativeai as genai
from utils.logger import logger



# ====== INIT CLIENTS ======
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ====== TEST CONNECTION ======
logger.info(f"Available indexes: {pc.list_indexes()}")

# ====== EMBEDDING FUNCTION ======
def embed_text(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return result["embedding"]


# ====== CHUNKING FUNCTION ======
def chunk_text(text, chunk_size=2000, overlap=500):
    """Split long text into larger chunks with overlap for faster embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ====== JURISDICTION EXTRACTION ======
def extract_jurisdiction(file_name, text=None):
    """
    Extract jurisdiction/country from file name or text. Treat 'EU' as a country.
    Extend this mapping as needed.
    """
    JURISDICTION_KEYWORDS = {
        "FCA": "UK", "UK": "UK", "MAS": "Singapore", "Singapore": "Singapore",
        "SEC": "USA", "USA": "USA", "US": "USA", "FinCEN": "USA", "FINTRAC": "Canada",
        "Canada": "Canada", "FSA": "Japan", "Japan": "Japan", "AUSTRAC": "Australia",
        "Australia": "Australia", "SFC": "Hong Kong", "Hong Kong": "Hong Kong",
        "ADGM": "UAE", "UAE": "UAE", "EU": "EU", "European Union": "EU"
    }
    # Check file name
    for keyword, country in JURISDICTION_KEYWORDS.items():
        if keyword.lower() in file_name.lower():
            return country
    # Optionally check text
    if text:
        for keyword, country in JURISDICTION_KEYWORDS.items():
            if keyword.lower() in text.lower():
                return country
    return "Unknown"

# ====== BATCHING UTILS ======
BATCH_SIZE = 50  # You can adjust this as needed
def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# ====== SKIP ALREADY EMBEDDED FILES ======
SKIP_FILES = {"AC_Regulations.json", "ADGM_Regulations.json", "APRA_Compliance.json"}

# ====== LOOP THROUGH FILES ======
for file_name in os.listdir(DATA_FOLDER):
    if not file_name.endswith(".json") or file_name in SKIP_FILES:
        continue

    file_path = os.path.join(DATA_FOLDER, file_name)
    logger.info(f"\U0001F4C4 Processing {file_name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        if isinstance(data, list):
            for i, doc in enumerate(data):
                text = doc.get("content") or doc.get("text") or str(doc)
                if text.strip():
                    jurisdiction = extract_jurisdiction(file_name, text)
                    for j, sub_chunk in enumerate(chunk_text(text)):
                        chunks.append((f"{file_name}_{i}_{j}", sub_chunk, jurisdiction))
        elif isinstance(data, dict):
            for key, value in data.items():
                text = str(value)
                if text.strip():
                    jurisdiction = extract_jurisdiction(file_name, text)
                    for j, sub_chunk in enumerate(chunk_text(text)):
                        chunks.append((f"{file_name}_{key}_{j}", sub_chunk, jurisdiction))
        else:
            jurisdiction = extract_jurisdiction(file_name, str(data))
            for j, sub_chunk in enumerate(chunk_text(str(data))):
                chunks.append((f"{file_name}_{j}", sub_chunk, jurisdiction))

        total_uploaded = 0
        for batch in batch_iterable(chunks, BATCH_SIZE):
            vectors = []
            for _id, text, jurisdiction in batch:
                vector = embed_text(text)
                vectors.append({
                    "id": _id,
                    "values": vector,
                    "metadata": {
                        "source": file_name,
                        "text": text[:300],
                        "jurisdiction": jurisdiction
                    }
                })
            if vectors:
                index.upsert(vectors=vectors)
                total_uploaded += len(vectors)
                logger.info(f"✅ Uploaded {len(vectors)} records from {file_name} (batch)")
        logger.info(f"✅ Uploaded {total_uploaded} records from {file_name} (total)")

    except Exception as e:
        logger.error(f"❌ Error processing {file_name}: {e}")

logger.info("🎉 All files processed and uploaded!")
logger.info(f"📊 Pinecone index stats: {index.describe_index_stats()}")
