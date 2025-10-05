
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

    index = get_index()
    query_embedding = embed_model.embed_content(query)["embedding"]

    results = index.query(vector=query_embedding, top_k=50, include_metadata=True)
    matches = results["matches"]

    jurisdictions = detect_jurisdictions(query)
    if jurisdictions:
        jurisdictions_lower = [j.lower() for j in jurisdictions]
        def is_jurisdiction_match(match):
            meta = match.get("metadata", {})
            # Prefer explicit jurisdiction field
            if meta.get("jurisdiction", "").lower() in jurisdictions_lower:
                return True
            # Fallback: check in source or text
            source = meta.get("source", "").lower()
            text = meta.get("text", "").lower()
            return any(j in source or j in text for j in jurisdictions_lower)
        matches = sorted(matches, key=lambda m: not is_jurisdiction_match(m))
        logger.info(f"Jurisdictions {jurisdictions} detected in query. Prioritizing relevant chunks.")
    else:
        logger.info("No jurisdiction detected in query. Returning top chunks.")

    # Return top_k reranked chunks
    return [match["metadata"]["text"] for match in matches[:top_k]]
