import re
from utils.logger import logger
# Mapping of regulator/keyword to jurisdiction/country
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
    # Add more as needed
}

def detect_jurisdictions(query):
    """Detect all jurisdictions/countries from query using known keywords."""
    found = set()
    for keyword, country in JURISDICTION_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            found.add(country)
    return list(found) if found else None
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai


# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Init Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def enhance_query(query):
    """Enhance/rephrase query for better retrieval."""
    # Detect jurisdictions for more targeted expansion
    jurisdictions = detect_jurisdictions(query)
    if jurisdictions and len(jurisdictions) > 1:
        compare_note = f"This is a comparative query about: {', '.join(jurisdictions)}."
    else:
        compare_note = ""
    enhancement_prompt = f"""
You are an expert in information retrieval and prompt engineering. Your job is to rewrite, expand, and clarify the following user query so that it retrieves the most relevant, comprehensive, and contextually rich information from a knowledge base.

{compare_note}

Instructions:
- Expand the query to include synonyms, related terms, and all possible regulatory frameworks, agencies, and compliance requirements for each jurisdiction mentioned.
- If the query is a comparison, explicitly mention all relevant countries, regions, or regulatory bodies in the enhanced query.
- Add keywords for laws, agencies, and compliance topics (e.g., MiCA, SEC, FinCEN, AML, CFT, registration, licensing, reporting, tax, cybersecurity, etc.).
- Clarify any ambiguities, but do not change the core intent.

Original query: "{query}"

Enhanced, retrieval-optimized query (detailed, with synonyms, related laws, agencies, and clarifications):
"""
    logger.info("Enhancing query for better retrieval.")
    response = model.generate_content(enhancement_prompt)
    return response.text.strip()

def retrieve_context(query, top_k=3):
    """Retrieve top-k documents from Pinecone using Gemini embeddings."""
    logger.info(f"Retrieving context for query: {query}")
    # Enhance and expand the query for better retrieval
    enhanced_query = enhance_query(query)

    # Detect all jurisdictions from query
    jurisdictions = detect_jurisdictions(query)

    # Combine original and enhanced query for embedding
    combined_query = f"{query}\n{enhanced_query}"
    embed_model = genai.embed_content(model="models/embedding-001", content=combined_query)
    query_embedding = embed_model['embedding']

    # Retrieve more chunks for reranking
    results = index.query(
        vector=query_embedding,
        top_k=80,  # retrieve more for reranking
        include_metadata=True
    )

    # Rerank: prioritize chunks with matching jurisdictions in metadata['jurisdiction']
    matches = results["matches"]
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
        # Sort: matches with any jurisdiction first, then others
        matches = sorted(matches, key=lambda m: not is_jurisdiction_match(m))
        logger.info(f"Jurisdictions {jurisdictions} detected in query. Prioritizing relevant chunks.")
    else:
        logger.info("No jurisdiction detected in query. Returning top chunks.")

    # Return top 16 reranked chunks
    results["matches"] = matches[:16]
    return results

def build_prompt(query, retrieved_docs):
    """Combine query + retrieved docs into one prompt for Gemini."""
    context_texts = [doc['metadata']['text'] for doc in retrieved_docs['matches']]
    context = "\n\n".join(context_texts)

    prompt = f"""
You are a highly knowledgeable crypto compliance assistant specializing in cryptocurrency regulations across multiple countries: USA, Japan, UAE, Australia, Canada, Singapore, and Hong Kong.

Your job is to provide extremely detailed, comprehensive, and well-structured answers to user questions using ONLY the provided context. Always:

1. **Directly answer the question first** with the most relevant and specific information from the context.
2. **Summarize and explain requirements in your own words.** Do not instruct the user to read or review the guide. Instead, extract and explain all key points, steps, and obligations from the context so the user can understand and act on them.
3. **Expand with details:** Give 5-8 bullet points or 4-8 sentences, covering all relevant aspects, nuances, and examples from the context.
4. **Provide country-specific details** and mention applicable regulations, agencies, or frameworks for each country mentioned in the context.
5. **Clarify gaps:** If the context does not fully answer the question, clearly state: "_The context does not provide full details on this jurisdiction, but generally…_"
6. **Avoid vague statements:** Do not hedge; clearly summarize or explain.
7. **Format consistently:** Use bullets, bold key terms, and include references if available in the context.
8. **Never output system warnings or errors.** Only return the answer to the user query.
9. **At the end, briefly cite the source document(s) if relevant, e.g., 'Source: A GUIDE TO DIGITAL TOKEN OFFERINGS (MAS, 2020)'.**

Context:
{context}

Question: {query}

Answer in a detailed, structured, and actionable way, summarizing the requirements from the context.
"""
    return prompt

def rag_answer(query):
    """Main RAG pipeline."""
    logger.info(f"Running RAG pipeline for query: {query}")
    results = retrieve_context(query)
    prompt = build_prompt(query, results)
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    while True:
        user_query = input("\n🔎 Enter your question (or type 'exit' to quit): ")
        if user_query.lower() in ["exit", "quit"]:
            logger.info("👋 Exiting RAG chatbot.")
            break

        answer = rag_answer(user_query)
        print("\n🤖 Answer:\n", answer)
