import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import RAGAgent

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="A simple API for interacting with the RAG Agent",
    version="1.0.0"
)

# Initialize the RAG Agent
try:
    agent = RAGAgent()
    print("🤖 RAG Agent initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize RAG Agent: {e}")
    agent = None

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_length: Optional[int] = None

class QueryResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "🤖 RAG Agent API is running!",
        "status": "healthy",
        "agent_ready": agent is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy" if agent is not None else "error",
        "agent_initialized": agent is not None,
        "api_version": "1.0.0"
    }

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query using the RAG Agent
    
    - **query**: The question or prompt to send to the agent
    - **max_length**: Optional maximum length for the response
    """
    if agent is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG Agent is not initialized"
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400, 
            detail="Query cannot be empty"
        )
    
    try:
        response = agent.process_query(request.query.strip())
        
        return QueryResponse(
            response=response,
            success=True
        )
        
    except Exception as e:
        return QueryResponse(
            response="",
            success=False,
            error=str(e)
        )

# Simple GET endpoint for quick testing
@app.get("/ask/{query}")
async def ask_simple(query: str):
    """Simple GET endpoint for quick testing - just pass query in URL"""
    if agent is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG Agent is not initialized"
        )
    
    try:
        response = agent.process_query(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the FastAPI server"""
    print("🚀 Starting RAG Agent FastAPI server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📋 Interactive docs at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "simpkeapi:app",  # Update this if your file has a different name
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

if __name__ == "__main__":
    main()