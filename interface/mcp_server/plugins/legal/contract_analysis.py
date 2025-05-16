from typing import Dict, Any
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class ContractAnalysisInput(PluginInput):
    contract_text: str = Field(..., description="The contract text to analyze")
    jurisdiction: str = Field("US", description="Legal jurisdiction")
    analysis_type: str = Field("compliance", description="Type of analysis to perform")

class ContractAnalysisOutput(PluginOutput):
    issues: Dict[str, Any] = Field(..., description="Identified legal issues")
    recommendations: str = Field(..., description="Recommended actions")
    risk_score: float = Field(..., description="Overall risk score (0-1)")

class ContractAnalysisPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.legal_db = None
    
    async def initialize(self):
        # Load legal database or models
        from legal_models import load_contract_analyzer
        self.legal_db = await load_contract_analyzer()
        self.initialized = True
    
    async def execute(self, input_data: ContractAnalysisInput) -> ContractAnalysisOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        analysis = await self.legal_db.analyze(
            input_data.contract_text,
            jurisdiction=input_data.jurisdiction,
            analysis_type=input_data.analysis_type
        )
        
        return ContractAnalysisOutput(
            issues=analysis.get("issues", {}),
            recommendations=analysis.get("recommendations", ""),
            risk_score=analysis.get("risk_score", 0.0)
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Contract Analysis Plugin",
            "version": "1.0.0",
            "description": "Analyzes contracts for legal compliance and risks"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return ContractAnalysisInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return ContractAnalysisOutput
