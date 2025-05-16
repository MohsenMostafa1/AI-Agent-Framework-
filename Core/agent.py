from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    
class BaseAgent(ABC):
    def __init__(self, 
                 llm: Any,
                 memory: Any,
                 tools: List[Tool] = None,
                 config: Dict[str, Any] = {}):
        self.llm = llm
        self.memory = memory
        self.tools = tools or []
        self.config = config
        self._setup_function_calling()
        
    def _setup_function_calling(self):
        """Prepare the LLM for tool/function calling"""
        if hasattr(self.llm, 'bind_tools'):
            self.llm.bind_tools(self.tools)
    
    def add_tool(self, tool: Tool):
        """Register a new tool with the agent"""
        self.tools.append(tool)
        self._setup_function_calling()
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Base generation method to be implemented by subclasses"""
        pass
        
    def run_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a registered tool"""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        return getattr(self, f"tool_{tool_name}")(**params)
    
    def __call__(self, input_text: str) -> str:
        """Standard interface for the agent"""
        # First check if this should be a direct tool call
        if self._should_use_tool(input_text):
            tool_name, params = self._parse_tool_input(input_text)
            result = self.run_tool(tool_name, params)
            return self._format_tool_result(result)
        
        # Otherwise proceed with standard generation
        return self.generate(input_text)
    
    # Helper methods omitted for brevity...
