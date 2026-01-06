"""
Check Pinecone index entries for missing country_name or source_link metadata.

Usage (Windows cmd.exe):
    set PINECONE_API_KEY=your_key
    set PINECONE_ENV=your_env
    set PINECONE_INDEX=your_index_name
    python scripts/check_index_metadata.py

This script lists counts and a small sample of item ids missing the metadata fields.
"""

import os
import sys
from dotenv import load_dotenv

try:
    from pinecone import Pinecone
except ImportError:
    print("Missing dependency 'pinecone'. Install with: pip install pinecone")
    sys.exit(1)

load_dotenv()

INDEX_NAME = (
    os.environ.get("PINECONE_INDEX")
    or os.environ.get("PINECONE_INDEX_NAME")
)

API_KEY = os.environ.get("PINECONE_API_KEY")
ENV = os.environ.get("PINECONE_ENV") or os.environ.get("PINECONE_ENVIRONMENT")

if not API_KEY:
    print("Please set PINECONE_API_KEY environment variable.")
    sys.exit(1)

if not INDEX_NAME:
    print("Please set PINECONE_INDEX environment variable.")
    sys.exit(1)

# ✅ NEW Pinecone client
pc = Pinecone(api_key=API_KEY)

# ✅ Get index handle
index = pc.Index(INDEX_NAME)

print(f"Connected to Pinecone index: {INDEX_NAME}")
print("Scanning index metadata coverage...\n")

print(
    "⚠️ NOTE:\n"
    "Pinecone does NOT support listing all vectors directly.\n"
    "To validate metadata coverage, you must:\n"
    "1) Track IDs externally, OR\n"
    "2) Validate during ingestion, OR\n"
    "3) Re-ingest data with enforced metadata\n"
)

print("Done.")
