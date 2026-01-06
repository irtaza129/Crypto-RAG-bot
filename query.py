import re
import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from utils.logger import logger

# =============================
# Environment & Clients
# =============================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, GEMINI_API_KEY]):
    logger.error("Missing required environment variables: ensure PINECONE_API_KEY, PINECONE_INDEX_NAME, GEMINI_API_KEY are set")
    raise ValueError("❌ Missing required environment variables")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Log initialization
logger.info("Initialized services: Pinecone index='%s', Generative model='%s'", PINECONE_INDEX_NAME, "gemini-2.5-flash-lite")

conversation = []
conversation_jurisdiction = []
MAX_TURNS = 3

SUPPORTED_JURISDICTIONS = [
    "USA", "UK", "EU", "Singapore", "Japan",
    "Australia", "Canada", "UAE", "Hong Kong",
    "Global", "Switzerland"
]

# =============================
# Lightweight keyword hints (NOT mapping)
# =============================
JURISDICTION_HINTS = [
    "USA", "US", "U.S.", "United States", "SEC", "CFTC", "FinCEN", "OCC", "NYDFS",
    "UK", "United Kingdom", "FCA", "Britain",
    "EU", "European Union", "ESMA", "EBA", "MiCA",
    "Singapore", "MAS",
    "Japan", "FSA",
    "Australia", "AUSTRAC",
    "Canada", "FINTRAC",
    "UAE", "ADGM",
    "Hong Kong", "SFC",
    "Switzerland", "FINMA"
]

# =============================
# Utilities
# =============================
def meta_get(meta: dict, *keys, default=None):
    if not meta:
        return default
    lower = {k.lower(): k for k in meta}
    for k in keys:
        if k.lower() in lower:
            return meta.get(lower[k.lower()])
    return default

# =============================
# Jurisdiction Detection (LLM-centric)
# =============================
def extract_jurisdiction_hints(query: str):
    logger.debug("extract_jurisdiction_hints: analyzing query: %s", query)
    found = set()
    for h in JURISDICTION_HINTS:
        if re.search(rf"\b{re.escape(h)}\b", query, re.IGNORECASE):
            found.add(h)
    hints = list(found)
    logger.debug("extract_jurisdiction_hints: found hints: %s", hints)
    return hints

def normalize_jurisdictions_with_llm(query: str, hints: list):
    """
    LLM is the single authority.
    It maps cities, states, regulators, regions → canonical jurisdictions.
    """
    prompt = f"""
Normalize jurisdiction references into canonical jurisdiction names.

Rules:
- US, U.S., America, states, cities (e.g. New York, California) → USA
- FCA → UK
- ESMA, EBA, MiCA, EU member states → EU
- Cities or regions (e.g. London, Paris, Berlin, Sydney) → their country/region
- Keep canonical names unchanged
- Only return jurisdictions from this allowed list:
{SUPPORTED_JURISDICTIONS}

Input query:
"{query}"

Detected hints (may be incomplete):
{json.dumps(hints)}

Return ONLY JSON:
{{"normalized_jurisdictions": []}}
"""
    try:
        logger.debug("normalize_jurisdictions_with_llm: sending prompt to model. hints=%s", hints)
        resp = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        logger.debug("normalize_jurisdictions_with_llm: model response text: %s", text[:100])
        parsed = json.loads(text)

        result = []
        seen = set()
        for j in parsed.get("normalized_jurisdictions", []):
            if j not in seen:
                result.append(j)
                seen.add(j)

        logger.info("normalize_jurisdictions_with_llm: normalized to %s", result)
        return result
    except Exception as e:
        logger.warning("Jurisdiction normalization failed: %s", e)
        return []

def detect_jurisdictions(query: str):
    hints = extract_jurisdiction_hints(query)
    return normalize_jurisdictions_with_llm(query, hints)

# =============================
# Query Resolution
# =============================
def resolve_query(query, convo):
    if not convo:
        return query

    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in convo[-6:]
    )

    prompt = f"""
Rewrite the user query into a standalone question.

Conversation:
{history}

User query:
"{query}"

Return ONLY the rewritten query.
"""
    logger.debug("resolve_query: prompt sent to model for rewriting")
    resp = model.generate_content(prompt)
    rewritten = resp.text.strip()
    logger.info("resolve_query: rewritten query: %s", rewritten)
    return rewritten

# =============================
# Query Enhancement
# =============================
def enhance_query(query, jurisdictions):
    note = f"Comparative query across: {', '.join(jurisdictions)}." if len(jurisdictions) > 1 else ""

    prompt = f"""
You are an expert in crypto compliance retrieval.

Rewrite this query to maximize recall.
{note}

Original:
"{query}"
Only write the enhanced query in response nothing else
Enhanced:
"""
    logger.debug("enhance_query: sending query to model for enhancement (jurisdictions=%s)", jurisdictions)
    resp = model.generate_content(prompt)
    enhanced = resp.text.strip()
    logger.info("enhance_query: enhanced query length=%d", len(enhanced))
    # Log the enhanced query content (truncated to 500 chars to avoid overly large logs)
    logger.info("enhance_query: enhanced query (truncated 500 chars): %s", enhanced[:500])
    return enhanced

# =============================
# Retriever
# =============================
def retrieve_context_weighted(query, primary_jurisdictions, top_k_primary=10, top_k_global=5):
    logger.info("retrieve_context_weighted: retrieving for query='%s' primary_jurisdictions=%s", query, primary_jurisdictions)
    enhanced = enhance_query(query, primary_jurisdictions)
    combined = f"{query}\n{enhanced}"

    embedding = genai.embed_content(
        model="models/embedding-001",
        content=combined
    )["embedding"]

    logger.debug("retrieve_context_weighted: embedding length=%d", len(embedding))

    results = []
    # Ensure primary_res is always defined so logs won't reference an undefined variable
    primary_res = {"matches": []}

    if primary_jurisdictions:
        primary_res = index.query(
            vector=embedding,
            top_k=top_k_primary,
            include_metadata=True,
            filter={"country_name": {"$in": primary_jurisdictions}}
        )
        for m in primary_res.get("matches", []):
            m["__priority"] = 2
            results.append(m)

    logger.info("retrieve_context_weighted: primary results=%d", len(primary_res.get("matches", [])) if primary_res else 0)

    global_res = index.query(
        vector=embedding,
        top_k=top_k_global,
        include_metadata=True,
        filter={"country_name": "Global_Resources"}
    )

    for m in global_res.get("matches", []):
        m["__priority"] = 1
        results.append(m)

    logger.info("retrieve_context_weighted: global results=%d total_results=%d", len(global_res.get("matches", [])), len(results))

    return rerank_results(results)

def rerank_results(matches):
    ranked = []
    for m in matches:
        score = m.get("score", 0)
        priority = m.get("__priority", 0)
        final_score = score + (0.15 * priority)

        ranked.append((
            final_score,
            {
                "id": m.get("id"),
                "score": score,
                "metadata": m.get("metadata", {})
            }
        ))

    ranked.sort(key=lambda x: x[0], reverse=True)
    res = {"matches": [m for _, m in ranked]}
    logger.debug("rerank_results: returning %d matches; top_score=%.4f", len(res["matches"]), ranked[0][0] if ranked else 0)
    return res

# =============================
# Prompt Builder
# =============================
def build_prompt(query, retrieved):
    context_blocks = []
    chunks = []

    for m in retrieved.get("matches", []):
        meta = m.get("metadata", {}) or {}

        text = meta_get(meta, "chunk_text", "text", default="")
        country = meta_get(meta, "country_name", "jurisdiction", default="Unknown")
        link = meta_get(meta, "source_link", default="")
        score = m.get("score", 0.0)

        chunks.append({
            "id": m.get("id"),
            "text": text,
            "country_name": country,
            "jurisdiction": country,
            "source": link,
            "score": score
        })

        context_blocks.append(
            f"""
Jurisdiction: {country}
Source: {link}
Score: {score:.2f}
{text}
""".strip()
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a Crypto Compliance Assistant , You are created by Irtaza Ali and your domain scope are crypto compliance, regulations, and anything adjacent to that.

Use the context below as the primary source to give detailed answer.
You may use external and your own knowledge, but only if it is not contradicted by the context or if the retrieved context is insufficient.
Clearly decline queries that are not crypto compliance-related(e.g Regular finance compliances ) or adjacent to these topics(AML ,KYC ,Taxation, regulations and reporting)
If the query is about a EU member state or city/state of supported jurisdiction , use context of its relevant region/country to answer and reason that it applies on it (e.g SEC,CFTC regulatios apply on a query about new york also similiarly MiCA applies on Germany as it is EU member state).
Decline to answer if query contains unsupported jurisdictions which is not one of the supported jurisdictions: {', '.join(SUPPORTED_JURISDICTIONS)}
You can greet and introduce yourself in response while offering to help with crypto compliance questions ONLY IF the query is a greeting or small talk , dont introduce and greet when asked normal compliances question.
You can ask the user feedback on whether He is asking as individual or business entity if query doesnt clarify or indicates any of these to give tailored response if the query response varies for individuals and responses.
You can ask for Jurisdiction from user and inform about your supported jurisdictions if query doesnt clarify or indicates any jurisdiction to give tailored response.
Cite sources using LINKS ONLY (no file names) at the end of response(NOT in between).

===== CONTEXT =====
{context}
===== END CONTEXT =====

User question:
{query}

Answer clearly and structure by jurisdiction if applicable.
"""
    logger.info("build_prompt: built prompt with %d chunks", len(chunks))
    return prompt, chunks

# =============================
# RAG Pipeline
# =============================
def rag_answer(query, convo, convo_jur):
    logger.info("rag_answer: starting RAG for query: %s", query)
    resolved = resolve_query(query, convo)

    jurisdictions = detect_jurisdictions(resolved)
    if not jurisdictions:
        jurisdictions = convo_jur or ["Global"]

    if "Global" not in jurisdictions:
        jurisdictions.append("Global")

    primary_jur = [j for j in jurisdictions if j != "Global"]

    results = retrieve_context_weighted(
        resolved,
        primary_jur,
        top_k_primary=10,
        top_k_global=5
    )

    prompt, chunks = build_prompt(resolved, results)
    resp = model.generate_content(prompt)
    answer = resp.text.strip()

    logger.info("rag_answer: generated answer length=%d for query='%s'", len(answer), query)

    convo.append({"role": "user", "content": query})
    convo.append({"role": "assistant", "content": answer})
    convo[:] = convo[-MAX_TURNS * 2 :]

    return answer, chunks, convo, jurisdictions

# =============================
# CLI
# =============================
if __name__ == "__main__":
    while True:
        q = input("\nAsk (or 'exit'): ")
        if q.lower() in {"exit", "quit"}:
            break

        ans, chunks, conversation, conversation_jurisdiction = rag_answer(
            q, conversation, conversation_jurisdiction
        )

        print("\n Answer:\n", ans)
        print(f"\n Retrieved Chunks ({len(chunks)}):")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] {c['country_name']} | {c['source']} | score={c['score']:.2f}")
            print(c["text"][:300], "...")
