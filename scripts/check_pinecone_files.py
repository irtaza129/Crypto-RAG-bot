"""
Diagnostic script to check which source files are actually in Pinecone index.
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from collections import defaultdict

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print(f"Scanning index '{PINECONE_INDEX_NAME}' for source files...\n")

# Query for all vectors and check their metadata
file_counts = defaultdict(int)
sample_ids = defaultdict(list)

try:
    # Do a broad query to get a sample of vectors
    # Using a dummy vector to get many results
    dummy_vector = [0.0] * 768  # Assuming 768-dim embeddings
    
    results = index.query(
        vector=dummy_vector,
        top_k=1000,
        include_metadata=True
    )
    
    matches = results.get("matches", [])
    
    for match in matches:
        meta = match.get("metadata", {}) or {}
        source = None
        
        # Try different metadata key variations
        for key in ["source", "source_name", "Source", "SOURCE"]:
            if key in meta:
                source = meta[key]
                break
        
        if source:
            file_counts[source] += 1
            if len(sample_ids[source]) < 3:
                sample_ids[source].append(match.get("id", ""))
    
    print("Files found in Pinecone index:")
    print("-" * 60)
    for file, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {file}: {count} chunks")
        if sample_ids[file]:
            print(f"    Sample IDs: {sample_ids[file][:2]}")
    
    print(f"\nTotal vectors found: {len(matches)}")
    print(f"Unique files: {len(file_counts)}")
    
except Exception as e:
    print(f"Error scanning index: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: try to fetch index stats
    try:
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
    except Exception as e2:
        print(f"Could not get stats: {e2}")
