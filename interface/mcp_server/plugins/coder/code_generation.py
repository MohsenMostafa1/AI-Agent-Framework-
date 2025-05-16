from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class CodeGenInput(PluginInput):
    requirements: str = Field(..., description="Code requirements")
    language: str = Field("python", description="Programming language")
    style: str = Field("standard", description="Coding style")
    examples: List[str] = Field([], description="Example code snippets")

class CodeGenOutput(PluginOutput):
    generated_code: str = Field(..., description="Generated code")
    tests: str = Field("", description="Generated test cases")
    explanation: str = Field("", description="Code explanation")

class CodeGenerationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.code_model = None
    
    async def initialize(self):
        from coder_models import CodeGenerator
        self.code_model = CodeGenerator(self.config)
        await self.code_model.load()
        self.initialized = True
    
    async def execute(self, input_data: CodeGenInput) -> CodeGenOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        result = await self.code_model.generate(
            input_data.requirements,
            input_data.language,
            input_data.style,
            input_data.examples
        )
        
        return CodeGenOutput(
            generated_code=result.get("code", ""),
            tests=result.get("tests", ""),
            explanation=result.get("explanation", "")
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Code Generation Plugin",
            "version": "1.0.0",
            "description": "Generates code based on requirements"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return CodeGenInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return CodeGenOutput
