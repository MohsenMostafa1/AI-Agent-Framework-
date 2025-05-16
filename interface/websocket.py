import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
from ..core.llm import QuantizedLLM
from ..core.mcp_integration import MCPHandler

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

async def handle_websocket(websocket: WebSocket, client_id: str):
    llm = QuantizedLLM.load_from_config()
    mcp = MCPHandler()
    
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Stream tokens one by one
            async for token in llm.stream_generate(data):
                await manager.send_personal_message(token, client_id)
                
                # Check for MCP interrupts
                if mcp.check_interrupt(client_id):
                    await manager.send_personal_message("[MCP_INTERRUPT]", client_id)
                    break
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
