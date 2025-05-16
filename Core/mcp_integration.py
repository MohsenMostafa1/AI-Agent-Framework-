import requests
from typing import Dict, Any, Optional
from pydantic import BaseModel

class MCPMessage(BaseModel):
    role: str  # "user", "agent", "system", "tool"
    content: str
    metadata: Dict[str, Any] = {}

class MCPClient:
    def __init__(self, 
                 base_url: str = "https://api.mcp.ai/v1",
                 api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        
    def send(self, 
             messages: List[MCPMessage],
             model_id: str = "default",
             stream: bool = False) -> MCPMessage:
        """Send a message to MCP endpoint"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": [msg.dict() for msg in messages],
            "model_id": model_id,
            "stream": stream
        }
        
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            stream=stream
        )
        
        if stream:
            return self._handle_stream(response)
        else:
            return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> MCPMessage:
        """Handle non-streaming response"""
        if response.status_code != 200:
            raise ValueError(f"MCP request failed: {response.text}")
        data = response.json()
        return MCPMessage(**data['choices'][0]['message'])
    
    def _handle_stream(self, response: requests.Response):
        """Handle streaming response"""
        # Implementation for token-by-token streaming
        pass
