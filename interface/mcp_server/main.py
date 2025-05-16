import uvicorn
from fastapi import FastAPI
from .plugins import PluginManager
from ..core.mcp_integration import MCPProtocol

app = FastAPI(title="MCP Server")
plugin_manager = PluginManager()
mcp_protocol = MCPProtocol()

@app.post("/mcp/v1/execute")
async def execute_mcp_command(command: dict):
    """Main MCP endpoint for model coordination"""
    # Validate protocol version
    if not mcp_protocol.validate(command):
        return {"error": "Invalid MCP request"}
    
    # Check for plugin routing
    if "plugin" in command:
        return await plugin_manager.execute(command["plugin"], command)
    
    # Standard MCP processing
    return await mcp_protocol.process(command)

@app.get("/mcp/plugins")
async def list_plugins():
    """List available MCP plugins"""
    return {"plugins": plugin_manager.list_plugins()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
