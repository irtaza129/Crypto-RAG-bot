from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query import rag_answer  # make sure this returns both answer and chunks
import uvicorn
import os

# Create app
app = FastAPI(
    title="Crypto Compliance RAG API",
    description="Ask regulatory questions and get detailed, jurisdiction-aware answers.",
    version="1.0"
)

# ✅ Allow frontend to call API from any domain (important for Render + Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and Response Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]

# Endpoint
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        answer, retrieved_chunks = rag_answer(request.query)
        return QueryResponse(answer=answer, retrieved_chunks=retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
