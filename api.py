from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query import rag_answer
import uvicorn

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

app = FastAPI(title="Crypto Compliance RAG API", description="Ask regulatory questions and get detailed, jurisdiction-aware answers.", version="1.0")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        answer = rag_answer(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)