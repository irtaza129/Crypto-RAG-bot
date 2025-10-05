
import google.generativeai as genai
from config.settings import GEMINI_API_KEY
from utils.logger import logger

genai.configure(api_key=GEMINI_API_KEY)

def generate_answer(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""
    You are a helpful assistant. Use ONLY the following context to answer the query.
    If you don’t know the answer from the context, say "I don’t know."

    Context:
    {context_text}

    Query:
    {query}
    """
    logger.info("Generating answer using Gemini for the provided query and context.")
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text
