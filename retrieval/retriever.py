
from vectorstore.pinecone_store import get_index
import google.generativeai as genai
from config.settings import GEMINI_API_KEY
from utils.logger import logger

genai.configure(api_key=GEMINI_API_KEY)
embed_model = genai.get_model("embedding-001")

def detect_jurisdictions(query):
    # Simple mapping, should match upload_json.py
    JURISDICTION_KEYWORDS = {
        "FCA": "UK", "UK": "UK", "MAS": "Singapore", "Singapore": "Singapore",
        "SEC": "USA", "USA": "USA", "US": "USA", "FinCEN": "USA", "FINTRAC": "Canada",
        "Canada": "Canada", "FSA": "Japan", "Japan": "Japan", "AUSTRAC": "Australia",
        "Australia": "Australia", "SFC": "Hong Kong", "Hong Kong": "Hong Kong",
        "ADGM": "UAE", "UAE": "UAE", "EU": "EU", "European Union": "EU"
    }
    found = set()
    for keyword, country in JURISDICTION_KEYWORDS.items():
        if keyword.lower() in query.lower():
            found.add(country)
    return list(found) if found else None

from vectorstore.pinecone_store import get_index
import google.generativeai as genai
from config.settings import GEMINI_API_KEY
from utils.logger import logger

genai.configure(api_key=GEMINI_API_KEY)


def detect_jurisdictions(query: str) -> list:
    """Detect common jurisdiction keywords in the query and return countries list."""
    JURISDICTION_KEYWORDS = {
        "FCA": "UK", "UK": "UK", "MAS": "Singapore", "Singapore": "Singapore",
        "SEC": "USA", "USA": "USA", "US": "USA", "FinCEN": "USA", "FINTRAC": "Canada",
        "Canada": "Canada", "FSA": "Japan", "Japan": "Japan", "AUSTRAC": "Australia",
        "Australia": "Australia", "SFC": "Hong Kong", "Hong Kong": "Hong Kong",
        "ADGM": "UAE", "UAE": "UAE", "EU": "EU", "European Union": "EU"
    }
    found = set()
    q_lower = (query or "").lower()
    for keyword, country in JURISDICTION_KEYWORDS.items():
        if keyword.lower() in q_lower:
            found.add(country)
    return list(found)


def retrieve_context(query: str, top_k: int = 16) -> dict:
    """Retrieve top-k matches from Pinecone for the given query.

    Returns the raw results dict from Pinecone with `matches` trimmed to `top_k`.
    The function will prioritise chunks whose metadata indicates the detected
    jurisdictions from the query.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    index = get_index()

    logger.info("Embedding query for retrieval")
    embed = genai.embed_content(model="models/embedding-001", content=query)
    query_embedding = embed.get("embedding")
    if not query_embedding:
        raise RuntimeError("Failed to produce query embedding")

    results = index.query(vector=query_embedding, top_k=50, include_metadata=True)
    matches = results.get("matches", [])

    # Helper: case-insensitive metadata getter
    def meta_get(meta: dict, *names, default=None):
        if not meta:
            return default
        lower_map = {k.lower(): k for k in meta.keys()}
        for name in names:
            key = lower_map.get(name.lower())
            if key is not None:
                return meta.get(key)
        return default

    # Re-rank by jurisdiction match if present
    jurisdictions = detect_jurisdictions(query)
    if jurisdictions:
        jurisdictions_lower = [j.lower() for j in jurisdictions]

        def is_jurisdiction_match(match: dict) -> bool:
            meta = match.get("metadata", {}) or {}
            # Prefer explicit jurisdiction or country_name (case-insensitive)
            jur = (meta_get(meta, "jurisdiction", "country_name", "country") or "").lower()
            if jur in jurisdictions_lower:
                return True
            # Fallback: check in source, source_link, or text fields
            source = (meta_get(meta, "source") or "") or ""
            source_link = (meta_get(meta, "source_link", "sourceLink", "source_url") or "") or ""
            text = (meta_get(meta, "text") or "") or ""
            src_text = f"{source} {source_link} {text}".lower()
            return any(j in src_text for j in jurisdictions_lower)

        # Prefer exact country_name / jurisdiction matches first
        preferred = [m for m in matches if (
            str(meta_get((m.get("metadata", {}) or {}), "country_name", "country") or "").lower() in jurisdictions_lower
            or str(meta_get((m.get("metadata", {}) or {}), "jurisdiction") or "").lower() in jurisdictions_lower
        )]

        if preferred:
            other = [m for m in matches if m not in preferred]
            matches = preferred + other
            logger.info(f"Jurisdictions {jurisdictions} detected in query. Returning {len(preferred)} preferred chunks first.")
        else:
            matches = sorted(matches, key=lambda m: not is_jurisdiction_match(m))
            logger.info(f"Jurisdictions {jurisdictions} detected in query. Prioritizing relevant chunks by fuzzy match.")
    else:
        logger.info("No jurisdiction detected in query. Returning top chunks.")

    results["matches"] = matches[:top_k]
    return results


def get_text_chunks(query: str, top_k: int = 16) -> list:
    """Convenience wrapper returning a list of text chunks (metadata.text)"""
    results = retrieve_context(query, top_k=top_k)
    texts = [m.get("metadata", {}).get("text", "") for m in results.get("matches", [])]
    return texts


__all__ = ["detect_jurisdictions", "retrieve_context", "get_text_chunks"]
