from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class FraudDetectionInput(PluginInput):
    transaction_data: Dict[str, Any] = Field(..., description="Transaction data to analyze")
    historical_data: List[Dict[str, Any]] = Field([], description="Historical transaction data")
    user_profile: Dict[str, Any] = Field({}, description="User profile information")

class FraudDetectionOutput(PluginOutput):
    is_fraud: bool = Field(..., description="Fraud prediction")
    confidence: float = Field(..., description="Confidence score (0-1)")
    indicators: List[str] = Field(..., description="List of fraud indicators")
    risk_score: float = Field(..., description="Overall risk score (0-1)")

class FraudDetectionPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
    
    async def initialize(self):
        from finance_models import load_fraud_model
        self.model = await load_fraud_model()
        self.initialized = True
    
    async def execute(self, input_data: FraudDetectionInput) -> FraudDetectionOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        prediction = await self.model.predict(
            input_data.transaction_data,
            input_data.historical_data,
            input_data.user_profile
        )
        
        return FraudDetectionOutput(
            is_fraud=prediction['is_fraud'],
            confidence=prediction['confidence'],
            indicators=prediction['indicators'],
            risk_score=prediction['risk_score']
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Fraud Detection Plugin",
            "version": "1.0.0",
            "description": "Detects potential fraudulent financial transactions"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return FraudDetectionInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return FraudDetectionOutput
