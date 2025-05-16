from typing import List, Dict
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class RecommendationInput(PluginInput):
    user_id: str = Field(..., description="User identifier")
    product_history: List[str] = Field([], description="User's product history")
    context: Dict[str, Any] = Field({}, description="Contextual information")

class RecommendationOutput(PluginOutput):
    recommendations: List[Dict[str, Any]] = Field(..., description="Recommended products")
    rationale: str = Field(..., description="Explanation for recommendations")
    confidence_scores: List[float] = Field([], description="Confidence scores for each recommendation")

class ProductRecommendationsPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.recommender = None
    
    async def initialize(self):
        from ecommerce_models import RecommendationEngine
        self.recommender = RecommendationEngine(self.config)
        await self.recommender.load_model()
        self.initialized = True
    
    async def execute(self, input_data: RecommendationInput) -> RecommendationOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        recommendations = await self.recommender.generate(
            input_data.user_id,
            input_data.product_history,
            input_data.context
        )
        
        return RecommendationOutput(
            recommendations=recommendations.get("items", []),
            rationale=recommendations.get("rationale", ""),
            confidence_scores=recommendations.get("scores", [])
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Product Recommendations Plugin",
            "version": "1.0.0",
            "description": "Generates personalized product recommendations for e-commerce"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return RecommendationInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return RecommendationOutput
