r"""Inspect a Pinecone vector's metadata by id.

Usage (Windows cmd.exe):
    set PINECONE_API_KEY=your_key
    set PINECONE_INDEX_NAME=your_index
    python scripts/inspect_vector.py CRS_TBD.json_79_23

This prints the raw metadata and a normalized view with lowercase keys to help debug key casing.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import json
try:
    from pinecone import Pinecone
except Exception:
    print("Please install pinecone-client: pip install pinecone-client")
    raise

if len(sys.argv) < 2:
    print("Usage: python scripts/inspect_vector.py <vector_id>")
    sys.exit(1)

vector_id = sys.argv[1]
API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
ENV = os.environ.get("PINECONE_ENV")

if not API_KEY or not INDEX_NAME:
    print("Please set PINECONE_API_KEY and PINECONE_INDEX_NAME environment variables.")
    sys.exit(1)

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

try:
    # Newer pinecone clients accept include_metadata, older ones do not.
    try:
        resp = index.fetch(ids=[vector_id], include_metadata=True)
    except TypeError:
        resp = index.fetch(ids=[vector_id])
    except Exception as e:
        # Some clients raise a different error message for unexpected params
        if "unexpected parameter" in str(e).lower():
            resp = index.fetch(ids=[vector_id])
        else:
            print(f"Error fetching vector: {e}")
            sys.exit(1)
except Exception as e:
    print(f"Error fetching vector: {e}")
    sys.exit(1)

def _extract_item_from_resp(resp_obj, vid):
    """Try multiple patterns to extract a vector item for id vid from resp_obj."""
    # 1) dict-like response
    try:
        if isinstance(resp_obj, dict):
            vectors = resp_obj.get('vectors')
            if isinstance(vectors, dict):
                return vectors.get(vid)
            if isinstance(vectors, list):
                for v in vectors:
                    if (isinstance(v, dict) and v.get('id') == vid) or (hasattr(v, 'id') and getattr(v, 'id') == vid):
                        return v
    except Exception:
        pass

    # 2) object with attribute .vectors (mapping-like or iterable)
    try:
        vectors = getattr(resp_obj, 'vectors', None)
        if vectors is not None:
            # mapping-like
            try:
                if hasattr(vectors, 'get'):
                    v = vectors.get(vid)
                    if v:
                        return v
            except Exception:
                pass
            # iterable
            try:
                for v in vectors:
                    if (isinstance(v, dict) and v.get('id') == vid) or (hasattr(v, 'id') and getattr(v, 'id') == vid):
                        return v
            except Exception:
                pass
    except Exception:
        pass

    # 3) try resp_obj['vectors'] access guarded
    try:
        vectors = resp_obj['vectors']
        if isinstance(vectors, dict):
            return vectors.get(vid)
        if isinstance(vectors, list):
            for v in vectors:
                if (isinstance(v, dict) and v.get('id') == vid) or (hasattr(v, 'id') and getattr(v, 'id') == vid):
                    return v
    except Exception:
        pass

    return None


item = _extract_item_from_resp(resp, vector_id)
if not item:
    # Could not find the vector in the returned shape — print resp for debugging
    try:
        print("\nFull response object:")
        print(json.dumps(resp, default=str, indent=2))
    except Exception:
        print(repr(resp))
    print(f"\nCould not extract vector with id '{vector_id}' from the response. Response shape above.")
    sys.exit(0)

metadata = item.get('metadata', {}) or {}
print("\nRaw metadata:")
print(json.dumps(metadata, indent=2))

# Normalized view
normalized = {k.lower(): v for k, v in metadata.items()}
print("\nNormalized keys (lowercased):")
print(json.dumps(normalized, indent=2))

# Helpful existence checks
print("\nChecks:")
for key in ("country_name", "country", "source_link", "source_name", "source"):
    print(f"- {key}:", key in normalized)

print("\nIf 'country_name' or 'source_link' are missing, you need to re-run ingestion or upsert that vector with the correct metadata.")
