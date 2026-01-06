"""
Delete the Pinecone index (if possible) or clear all vectors, then upload all JSON files.
Usage:
    python scripts\delete_and_upload_all.py
"""
import os
import sys
import pathlib
import subprocess
from dotenv import load_dotenv

# ensure repo root in path
repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pinecone import Pinecone, ServerlessSpec
from utils.logger import logger

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    logger.error("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME")
    sys.exit(1)

pc = Pinecone(api_key=PINECONE_API_KEY)

logger.info(f"Attempting to delete index '{PINECONE_INDEX_NAME}' if it exists...")
try:
    # Try admin delete index
    try:
        pc.delete_index(PINECONE_INDEX_NAME)
        logger.info(f"pc.delete_index('{PINECONE_INDEX_NAME}') succeeded")
    except Exception as e:
        logger.warning(f"pc.delete_index not available or failed: {e}. Will attempt to delete vectors instead.")
        # Fallback: delete all vectors from index
        ix = pc.Index(PINECONE_INDEX_NAME)
        try:
            ix.delete(delete_all=True)
            logger.info("Cleared all vectors from index using delete_all=True")
        except Exception as e2:
            try:
                ix.delete_all()
                logger.info("Cleared all vectors from index using delete_all()")
            except Exception as e3:
                logger.warning(f"Failed to clear vectors: {e3}")

except Exception as e:
    logger.error(f"Unexpected error while deleting index: {e}")

# Recreate index if necessary
try:
    if PINECONE_INDEX_NAME not in pc.list_indexes():
        dim = int(os.getenv('PINECONE_DIMENSION', '1536'))
        cloud = os.getenv('PINECONE_CLOUD', 'aws')
        region = os.getenv('PINECONE_REGION', 'us-east-1')
        spec = ServerlessSpec(cloud=cloud, region=region)
        try:
            pc.create_index(name=PINECONE_INDEX_NAME, dimension=dim, spec=spec)
            logger.info(f"Created index {PINECONE_INDEX_NAME} with dimension {dim}, cloud={cloud}, region={region}")
        except Exception as e:
            logger.warning(f"ServerlessSpec create failed: {e}")
            try:
                pc.create_index(name=PINECONE_INDEX_NAME, dimension=dim)
                logger.info(f"Created index {PINECONE_INDEX_NAME} with dimension {dim} (fallback)")
            except Exception as e2:
                logger.warning(f"All create_index attempts failed: {e2}")
    else:
        logger.info(f"Index {PINECONE_INDEX_NAME} exists or was recreated by provider")
except Exception as e:
    logger.warning(f"Could not create index: {e}")

# Now run upload_json.py to upload all files
logger.info("Starting upload_json.py to upload all files...")
cmd = [sys.executable, os.path.join(repo_root, 'upload_json.py')]
proc = subprocess.run(cmd)
if proc.returncode == 0:
    logger.info("upload_json.py completed successfully")
else:
    logger.warning(f"upload_json.py exited with code {proc.returncode}")

print('done')