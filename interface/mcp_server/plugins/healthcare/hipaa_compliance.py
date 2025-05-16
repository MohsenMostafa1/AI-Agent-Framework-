from typing import Dict, List, Any
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class HIPAAInput(PluginInput):
    document_text: str = Field(..., description="Text to check for HIPAA compliance")
    document_type: str = Field(..., description="Type of healthcare document")

class HIPAAOutput(PluginOutput):
    violations: List[Dict[str, Any]] = Field(..., description="List of HIPAA violations found")
    compliance_score: float = Field(..., description="Overall compliance score (0-1)")
    recommendations: List[str] = Field(..., description="Recommendations for compliance")

class HIPAACompliancePlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hipaa_checker = None
    
    async def initialize(self):
        from healthcare_models import load_hipaa_checker
        self.hipaa_checker = await load_hipaa_checker()
        self.initialized = True
    
    async def execute(self, input_data: HIPAAInput) -> HIPAAOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        result = await self.hipaa_checker.analyze(
            input_data.document_text,
            doc_type=input_data.document_type
        )
        
        return HIPAAOutput(
            violations=result.get("violations", []),
            compliance_score=result.get("score", 0.0),
            recommendations=result.get("recommendations", [])
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "HIPAA Compliance Plugin",
            "version": "1.0.0",
            "description": "Checks documents for HIPAA compliance issues"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return HIPAAInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return HIPAAOutput
