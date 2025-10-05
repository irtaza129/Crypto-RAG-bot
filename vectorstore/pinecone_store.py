
from pinecone import Pinecone, ServerlessSpec
from config.settings import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME
from utils.logger import logger

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_index():
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",    # or "gcp" if you chose that
                region=PINECONE_ENV
            )
        )
    else:
        logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)
