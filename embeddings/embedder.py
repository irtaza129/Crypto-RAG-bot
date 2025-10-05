
import google.generativeai as genai
from config.settings import GEMINI_API_KEY
from vectorstore.pinecone_store import get_index
from utils.logger import logger

genai.configure(api_key=GEMINI_API_KEY)
model = genai.get_model("embedding-001")

def embed_and_store(data_chunks):
    index = get_index()

    vectors = []
    for idx, chunk in enumerate(data_chunks):
        embedding = model.embed_content(chunk["text"])["embedding"]
        vectors.append((f"id-{idx}", embedding, {"text": chunk["text"], "meta": chunk["meta"]}))

    # Upsert into Pinecone
    index.upsert(vectors)
    logger.info("✅ Data stored in Pinecone")
