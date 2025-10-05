from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query import rag_answer  # make sure this returns both answer and chunks
import uvicorn

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]  # <-- include the chunks

app = FastAPI(
    title="Crypto Compliance RAG API",
    description="Ask regulatory questions and get detailed, jurisdiction-aware answers.",
    version="1.0"
)

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        answer, retrieved_chunks = rag_answer(request.query)  # must return both
        return QueryResponse(answer=answer, retrieved_chunks=retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
import os