import re
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from utils.logger import logger

# === Load environment variables ===
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("❌ Missing API keys! Make sure .env is loaded correctly.")

# === Initialize clients ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")


# === Jurisdiction Detection ===
JURISDICTION_MAP = {
    "FCA": "UK",
    "UK": "UK",
    "MAS": "Singapore",
    "Singapore": "Singapore",
    "SEC": "USA",
    "USA": "USA",
    "US": "USA",
    "FinCEN": "USA",
    "FINTRAC": "Canada",
    "Canada": "Canada",
    "FSA": "Japan",
    "Japan": "Japan",
    "AUSTRAC": "Australia",
    "Australia": "Australia",
    "SFC": "Hong Kong",
    "Hong Kong": "Hong Kong",
    "ADGM": "UAE",
    "UAE": "UAE",
    "EU": "EU",
    "European Union": "EU",
}

def detect_jurisdictions(query):
    """Detect all jurisdictions/countries from query using known keywords."""
    found = set()
    for keyword, country in JURISDICTION_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            found.add(country)
    return list(found) if found else None


# === Query Enhancement ===
def enhance_query(query):
    jurisdictions = detect_jurisdictions(query)
    if jurisdictions and len(jurisdictions) > 1:
        compare_note = f"This is a comparative query about: {', '.join(jurisdictions)}."
    else:
        compare_note = ""

    enhancement_prompt = f"""
You are an expert in information retrieval and prompt engineering. Rewrite and expand the query for maximum recall and relevance.
{compare_note}

Original query: "{query}"

Enhanced, retrieval-optimized query:
"""
    logger.info("Enhancing query for better retrieval.")
    response = model.generate_content(enhancement_prompt)
    return response.text.strip()


# === Retriever ===
def retrieve_context(query, top_k=3):
    """Retrieve top-k documents from Pinecone using Gemini embeddings."""
    logger.info(f"Retrieving context for query: {query}")
    enhanced_query = enhance_query(query)
    jurisdictions = detect_jurisdictions(query)
    combined_query = f"{query}\n{enhanced_query}"

    embed_model = genai.embed_content(model="models/embedding-001", content=combined_query)
    query_embedding = embed_model["embedding"]

    results = index.query(vector=query_embedding, top_k=80, include_metadata=True)
    matches = results["matches"]

    if jurisdictions:
        jurisdictions_lower = [j.lower() for j in jurisdictions]

        def is_jurisdiction_match(match):
            meta = match.get("metadata", {})
            if meta.get("jurisdiction", "").lower() in jurisdictions_lower:
                return True
            source = meta.get("source", "").lower()
            text = meta.get("text", "").lower()
            return any(j in source or j in text for j in jurisdictions_lower)

        matches = sorted(matches, key=lambda m: not is_jurisdiction_match(m))
        logger.info(f"Jurisdictions {jurisdictions} detected. Prioritizing relevant chunks.")
    else:
        logger.info("No jurisdiction detected. Returning top chunks.")

    results["matches"] = matches[:16]
    return results


# === Prompt Builder ===
def build_prompt(query, retrieved_docs):
    context_texts = [doc["metadata"]["text"] for doc in retrieved_docs["matches"]]
    context = "\n\n".join(context_texts)

    prompt = f"""
You are a highly knowledgeable crypto compliance assistant specializing in cryptocurrency regulations worldwide.

Context:
{context}

Question: {query}

Answer in a detailed, structured, and actionable way.
"""
    return prompt, context_texts


# === Main RAG Pipeline ===
def rag_answer(query):
    """Main RAG pipeline returning both answer and retrieved chunks."""
    logger.info(f"Running RAG pipeline for query: {query}")

    results = retrieve_context(query)
    prompt, context_chunks = build_prompt(query, results)

    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Return both answer + retrieved chunks
    return answer, context_chunks


# === CLI for Local Testing ===
if __name__ == "__main__":
    while True:
        user_query = input("\n🔎 Enter your question (or type 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            logger.info("👋 Exiting RAG chatbot.")
            break
        try:
            answer, chunks = rag_answer(user_query)
            print("\n🤖 Answer:\n", answer)
            print("\n📚 Retrieved Chunks (Context):")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n[{i}] {chunk[:500]}...")
        except Exception as e:
            logger.error(f"Error: {e}")
            print("❌ Error while processing query:", e)
