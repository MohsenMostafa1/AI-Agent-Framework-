from typing import List, Dict, Optional
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field
from datetime import date

class SECFilingsInput(PluginInput):
    company: str = Field(..., description="Company name or ticker symbol")
    filing_type: str = Field("10-K", description="Type of SEC filing")
    year: Optional[int] = Field(None, description="Specific year to retrieve")
    quarter: Optional[int] = Field(None, description="Quarter for quarterly filings")

class SECFilingsOutput(PluginOutput):
    filings: List[Dict[str, Any]] = Field(..., description="List of SEC filings")
    metrics: Dict[str, Any] = Field({}, description="Financial metrics extracted")
    risk_factors: List[str] = Field([], description="Identified risk factors")

class SECFilingsPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sec_client = None
        self.analyzer = None
    
    async def initialize(self):
        from finance_models import SECClient, FilingAnalyzer
        self.sec_client = SECClient()
        self.analyzer = await FilingAnalyzer.load()
        self.initialized = True
    
    async def execute(self, input_data: SECFilingsInput) -> SECFilingsOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        filings = await self.sec_client.get_filings(
            input_data.company,
            input_data.filing_type,
            year=input_data.year,
            quarter=input_data.quarter
        )
        
        analysis = await self.analyzer.analyze_filings(filings)
        
        return SECFilingsOutput(
            filings=filings,
            metrics=analysis.get("metrics", {}),
            risk_factors=analysis.get("risk_factors", [])
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "SEC Filings Plugin",
            "version": "1.0.0",
            "description": "Retrieves and analyzes SEC filings for companies"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return SECFilingsInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return SECFilingsOutput
