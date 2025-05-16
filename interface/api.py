from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ..core.agent import BaseAgent
from ..core.mcp_integration import validate_mcp_request
import json

app = FastAPI(title="AI Agent Framework API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/completions")
async def chat_completion(request: dict):
    """Endpoint compatible with OpenAI API format"""
    try:
        validate_mcp_request(request)  # MCP validation
        
        agent = BaseAgent.from_config()
        
        def stream_generator():
            for chunk in agent.stream_response(request["messages"]):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/tools/execute")
async def execute_tool(request: dict):
    """For function calling"""
    agent = BaseAgent.from_config()
    return await agent.execute_tool(request)

@app.get("/mcp/status")
async def mcp_status():
    """Check MCP server connectivity"""
    return {"status": "active"}
