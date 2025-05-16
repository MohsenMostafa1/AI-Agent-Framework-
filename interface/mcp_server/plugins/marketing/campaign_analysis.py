from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class CampaignInput(PluginInput):
    campaign_data: Dict[str, Any] = Field(..., description="Campaign performance data")
    goals: Dict[str, Any] = Field({}, description="Campaign goals")
    historical_data: List[Dict[str, Any]] = Field([], description="Historical campaign data")

class CampaignOutput(PluginOutput):
    performance_metrics: Dict[str, Any] = Field(..., description="Performance analysis")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    roi_estimate: float = Field(..., description="Estimated ROI")

class CampaignAnalysisPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.analyzer = None
    
    async def initialize(self):
        from marketing_models import CampaignAnalyzer
        self.analyzer = CampaignAnalyzer(self.config)
        await self.analyzer.load_model()
        self.initialized = True
    
    async def execute(self, input_data: CampaignInput) -> CampaignOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        analysis = await self.analyzer.analyze(
            input_data.campaign_data,
            input_data.goals,
            input_data.historical_data
        )
        
        return CampaignOutput(
            performance_metrics=analysis.get("metrics", {}),
            recommendations=analysis.get("recommendations", []),
            roi_estimate=analysis.get("roi", 0.0)
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Campaign Analysis Plugin",
            "version": "1.0.0",
            "description": "Analyzes marketing campaign performance"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return CampaignInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return CampaignOutput
