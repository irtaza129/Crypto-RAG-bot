from flask import Blueprint, request, jsonify
from retrieval.retriever import retrieve
from generation.generator import generate_answer

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")

    # Step 1: retrieve docs
    context_chunks = retrieve(query, top_k=8)

    # Step 2: generate answer
    answer = generate_answer(query, context_chunks)

    return jsonify({"query": query, "answer": answer})
