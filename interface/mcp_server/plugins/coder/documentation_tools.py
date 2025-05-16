from typing import Dict
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class DocumentationInput(PluginInput):
    code: str = Field(..., description="Code to document")
    language: str = Field("python", description="Programming language")
    style: str = Field("standard", description="Documentation style")

class DocumentationOutput(PluginOutput):
    documentation: str = Field(..., description="Generated documentation")
    examples: List[str] = Field([], description="Usage examples")
    api_reference: Dict[str, Any] = Field({}, description="API reference")

class DocumentationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.doc_generator = None
    
    async def initialize(self):
        from coder_models import DocumentationGenerator
        self.doc_generator = DocumentationGenerator(self.config)
        await self.doc_generator.load_model()
        self.initialized = True
    
    async def execute(self, input_data: DocumentationInput) -> DocumentationOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        docs = await self.doc_generator.generate(
            input_data.code,
            input_data.language,
            input_data.style
        )
        
        return DocumentationOutput(
            documentation=docs.get("documentation", ""),
            examples=docs.get("examples", []),
            api_reference=docs.get("api_reference", {})
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Documentation Generator Plugin",
            "version": "1.0.0",
            "description": "Generates documentation from code"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return DocumentationInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return DocumentationOutput
