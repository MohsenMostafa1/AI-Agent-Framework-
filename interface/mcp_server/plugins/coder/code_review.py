from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class CodeReviewInput(PluginInput):
    code: str = Field(..., description="Code to review")
    language: str = Field("python", description="Programming language")
    guidelines: Dict[str, Any] = Field({}, description="Coding guidelines")

class CodeReviewOutput(PluginOutput):
    issues: List[Dict[str, Any]] = Field(..., description="Identified issues")
    score: float = Field(..., description="Code quality score (0-1)")
    suggestions: List[str] = Field([], description="Improvement suggestions")

class CodeReviewPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.reviewer = None
    
    async def initialize(self):
        from coder_models import CodeReviewer
        self.reviewer = CodeReviewer(self.config)
        await self.reviewer.load_model()
        self.initialized = True
    
    async def execute(self, input_data: CodeReviewInput) -> CodeReviewOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        review = await self.reviewer.review(
            input_data.code,
            input_data.language,
            input_data.guidelines
        )
        
        return CodeReviewOutput(
            issues=review.get("issues", []),
            score=review.get("score", 0.0),
            suggestions=review.get("suggestions", [])
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Code Review Plugin",
            "version": "1.0.0",
            "description": "Performs automated code reviews"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return CodeReviewInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return CodeReviewOutput
