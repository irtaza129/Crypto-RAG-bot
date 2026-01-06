from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from query import rag_answer  # make sure this returns both answer and chunks
import uvicorn
import os
import json

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
class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    query: str
    conversation_history: list[ConversationTurn] = []  # Optional: previous Q&A pairs for context

class ChunkMetadata(BaseModel):
    id: str
    text: str
    source: str
    jurisdiction: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[ChunkMetadata]
    conversation_history: list[ConversationTurn]  # Full conversation including this new answer

# Unified Endpoint
@app.post("/query")
def query_endpoint(request: QueryRequest, stream: bool = Query(False)):
    """
    Query the RAG system with conversation context.
    
    Args:
        request: QueryRequest with:
            - 'query': Your question (required)
            - 'conversation_history': List of previous Q&A pairs (optional) for context
        stream: If True, stream the answer via Server-Sent Events (SSE)
               If False (default), return full response as JSON
    
    Returns:
        - answer: The RAG response
        - retrieved_chunks: Sources used
        - conversation_history: Full conversation including this new answer
    
    Example 1 (First question - no history):
        POST /query
        {
          "query": "What are UK FCA rules?",
          "conversation_history": []
        }
        
    Example 2 (Follow-up question - include previous Q&A):
        POST /query
        {
          "query": "Tell me more about penalties",
          "conversation_history": [
            {"role": "user", "content": "What are UK FCA rules?"},
            {"role": "assistant", "content": "The FCA has established..."}
          ]
        }
        
    Example 3 (Streaming with history):
        POST /query?stream=true
        {
          "query": "What else?",
          "conversation_history": [...]
        }
    """
    
    # Convert conversation history from request to format expected by rag_answer
    conversation_list = [{"role": turn.role, "content": turn.content} for turn in request.conversation_history]
    conversation_jurisdiction = []  # Will be populated by rag_answer
    
    try:
        if stream:
            # Return SSE stream
            def event_stream():
                try:
                    answer, retrieved_chunks, updated_conversation, updated_jurisdictions = rag_answer(
                        request.query, 
                        conversation_list, 
                        conversation_jurisdiction
                    )
                    
                    # Yield the answer in chunks so clients can render progressively.
                    # We use SSE 'data:' lines so browsers and SSE clients can parse.
                    chunk_size = 512
                    for i in range(0, len(answer), chunk_size):
                        part = answer[i : i + chunk_size]
                        # Each SSE message must end with a blank line
                        yield f"data: {part}\n\n"

                    # Send final metadata event with retrieved chunks and conversation history
                    conversation_turns = [{"role": turn["role"], "content": turn["content"]} for turn in updated_conversation]
                    metadata = {
                        "retrieved_chunks": retrieved_chunks,
                        "conversation_history": conversation_turns
                    }
                    yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

                except Exception as e:
                    # Send an error event so the client can handle it
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        else:
            # Return standard JSON response
            answer, retrieved_chunks, updated_conversation, updated_jurisdictions = rag_answer(
                request.query,
                conversation_list,
                conversation_jurisdiction
            )
            
            # Convert chunk dicts to ChunkMetadata objects
            chunks_meta = [ChunkMetadata(**chunk) for chunk in retrieved_chunks]
            
            # Convert conversation history to ConversationTurn objects
            conversation_turns = [ConversationTurn(role=turn["role"], content=turn["content"]) for turn in updated_conversation]
            
            return QueryResponse(
                answer=answer, 
                retrieved_chunks=chunks_meta, 
                conversation_history=conversation_turns
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
