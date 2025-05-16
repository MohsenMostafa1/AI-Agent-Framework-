from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

class PluginInput(BaseModel):
    """Base input model for all plugins"""
    pass

class PluginOutput(BaseModel):
    """Base output model for all plugins"""
    pass

class BasePlugin(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize plugin resources"""
        pass
    
    @abstractmethod
    async def execute(self, input_data: PluginInput) -> PluginOutput:
        """Main execution method"""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return plugin metadata"""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> type[PluginInput]:
        """Return input schema class"""
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[PluginOutput]:
        """Return output schema class"""
        pass
