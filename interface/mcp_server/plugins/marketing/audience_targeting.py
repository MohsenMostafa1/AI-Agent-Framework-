from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class AudienceInput(PluginInput):
    customer_data: List[Dict[str, Any]] = Field(..., description="Customer data")
    product_data: Dict[str, Any] = Field({}, description="Product information")
    campaign_goals: Dict[str, Any] = Field({}, description="Marketing goals")

class AudienceOutput(PluginOutput):
    segments: Dict[str, List[str]] = Field(..., description="Target segments")
    targeting_rules: Dict[str, Any] = Field(..., description="Targeting rules")
    predicted_response: Dict[str, float] = Field(..., description="Predicted response rates")

class AudienceTargetingPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.targeting_model = None
    
    async def initialize(self):
        from marketing_models import AudienceTargeter
        self.targeting_model = AudienceTargeter(self.config)
        await self.targeting_model.load_model()
        self.initialized = True
    
    async def execute(self, input_data: AudienceInput) -> AudienceOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        targeting = await self.targeting_model.target(
            input_data.customer_data,
            input_data.product_data,
            input_data.campaign_goals
        )
        
        return AudienceOutput(
            segments=targeting.get("segments", {}),
            targeting_rules=targeting.get("rules", {}),
            predicted_response=targeting.get("response", {})
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Audience Targeting Plugin",
            "version": "1.0.0",
            "description": "Identifies and targets marketing audiences"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return AudienceInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return AudienceOutput
