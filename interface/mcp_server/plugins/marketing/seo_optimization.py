from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class SEOInput(PluginInput):
    content: str = Field(..., description="Content to optimize")
    keywords: List[str] = Field([], description="Target keywords")
    competitors: List[str] = Field([], description="Competitor URLs")

class SEOOutput(PluginOutput):
    optimized_content: str = Field(..., description="Optimized content")
    keyword_analysis: Dict[str, Any] = Field(..., description="Keyword analysis")
    recommendations: List[str] = Field(..., description="SEO recommendations")
    score: float = Field(..., description="SEO score (0-100)")

class SEOOptimizationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.seo_analyzer = None
    
    async def initialize(self):
        from marketing_models import SEOAnalyzer
        self.seo_analyzer = SEOAnalyzer(self.config)
        await self.seo_analyzer.load_model()
        self.initialized = True
    
    async def execute(self, input_data: SEOInput) -> SEOOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        analysis = await self.seo_analyzer.analyze(
            input_data.content,
            input_data.keywords,
            input_data.competitors
        )
        
        return SEOOutput(
            optimized_content=analysis.get("optimized_content", ""),
            keyword_analysis=analysis.get("keyword_analysis", {}),
            recommendations=analysis.get("recommendations", []),
            score=analysis.get("score", 0.0)
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "SEO Optimization Plugin",
            "version": "1.0.0",
            "description": "Optimizes content for search engines"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return SEOInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return SEOOutput
