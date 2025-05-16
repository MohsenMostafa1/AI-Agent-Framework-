import json
import requests
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
import inspect
from functools import wraps
import httpx
from urllib.parse import urlencode

# Constants
MCP_SEARCH_API = "https://api.mcp.ai/v1/search"
DEFAULT_HEADERS = {
    "User-Agent": "AI-Agent-Framework/1.0",
    "Accept": "application/json"
}

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None

class ToolSchema(BaseModel):
    """Schema for tool definition following MCP standard"""
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What the tool does")
    parameters: List[ToolParameter] = Field(default_factory=list)
    required_params: List[str] = Field(default_factory=list)
    endpoint: Optional[str] = Field(None, description="API endpoint if applicable")
    method: str = Field("GET", description="HTTP method if applicable")

class Tool(BaseModel):
    """Tool instance with executable function"""
    schema: ToolSchema
    function: Callable
    
    class Config:
        arbitrary_types_allowed = True
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        # Validate parameters against schema
        self._validate_params(kwargs)
        return self.function(**kwargs)
    
    def _validate_params(self, params: Dict[str, Any]):
        """Validate input parameters against tool schema"""
        for param in self.schema.parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    raise ValueError(f"Missing required parameter: {param.name}")
                params[param.name] = param.default
                
            if param.name in params:
                # Type checking would be more robust in production
                if param.enum and params[param.name] not in param.enum:
                    raise ValueError(f"Invalid value for {param.name}. Must be one of: {param.enum}")

class WebSearchTool:
    """MCP-standard web search implementation"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.schema = self._create_schema()
    
    def _create_schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="Perform web searches using MCP-standard API",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="integer",
                    description="Number of results to return (1-10)",
                    required=False,
                    default=5,
                    enum=[str(i) for i in range(1, 11)]
                ),
                ToolParameter(
                    name="freshness",
                    type="string",
                    description="Result freshness filter",
                    required=False,
                    enum=["day", "week", "month", "year"]
                )
            ],
            endpoint=MCP_SEARCH_API,
            method="GET"
        )
    
    async def search(self, 
                    query: str, 
                    num_results: int = 5,
                    freshness: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute web search with MCP API"""
        params = {
            "q": query,
            "limit": num_results
        }
        
        if freshness:
            params["freshness"] = freshness
            
        headers = DEFAULT_HEADERS.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    MCP_SEARCH_API,
                    params=params,
                    headers=headers,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()["results"]
        except httpx.RequestError as e:
            raise ConnectionError(f"Search API request failed: {str(e)}")
    
    def as_tool(self) -> Tool:
        """Convert this instance into a Tool object"""
        return Tool(
            schema=self.schema,
            function=self.search
        )

class ToolManager:
    """Central registry for all available tools"""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_core_tools()
    
    def _register_core_tools(self):
        """Register built-in tools"""
        web_search = WebSearchTool().as_tool()
        self.register_tool(web_search)
        
        # Register other core tools here
        self.register_tool(self._create_calculator_tool())
    
    def _create_calculator_tool(self) -> Tool:
        """Create a simple calculator tool"""
        schema = ToolSchema(
            name="calculator",
            description="Perform mathematical calculations",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate",
                    required=True
                )
            ]
        )
        
        def calculator(expression: str) -> float:
            # WARNING: In production, use a safe eval alternative
            try:
                return eval(expression, {"__builtins__": None}, {})
            except Exception as e:
                raise ValueError(f"Calculation failed: {str(e)}")
        
        return Tool(schema=schema, function=calculator)
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        if tool.schema.name in self._tools:
            raise ValueError(f"Tool '{tool.schema.name}' already registered")
        self._tools[tool.schema.name] = tool
    
    def register_function(self, 
                         func: Callable,
                         name: Optional[str] = None,
                         description: Optional[str] = None):
        """Decorator to register a function as a tool"""
        sig = inspect.signature(func)
        
        # Create parameters from function signature
        parameters = []
        required_params = []
        for param_name, param in sig.parameters.items():
            if
