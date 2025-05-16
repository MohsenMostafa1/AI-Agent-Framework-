from typing import List, Dict
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class ClinicalTermsInput(PluginInput):
    text: str = Field(..., description="Text to analyze for clinical terms")
    language: str = Field("en", description="Language of the text")

class ClinicalTermsOutput(PluginOutput):
    terms: List[Dict[str, Any]] = Field(..., description="Identified clinical terms")
    normalized: List[str] = Field(..., description="Normalized medical terms")
    icd_codes: List[str] = Field([], description="Related ICD codes")

class ClinicalTermsPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.term_extractor = None
        self.icd_mapper = None
    
    async def initialize(self):
        from healthcare_models import load_term_extractor, load_icd_mapper
        self.term_extractor = await load_term_extractor()
        self.icd_mapper = await load_icd_mapper()
        self.initialized = True
    
    async def execute(self, input_data: ClinicalTermsInput) -> ClinicalTermsOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        terms = await self.term_extractor.extract(input_data.text, input_data.language)
        normalized = [t['normalized'] for t in terms]
        icd_codes = await self.icd_mapper.map_terms(normalized)
        
        return ClinicalTermsOutput(
            terms=terms,
            normalized=normalized,
            icd_codes=icd_codes
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Clinical Terms Plugin",
            "version": "1.0.0",
            "description": "Extracts and normalizes clinical terms from text"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return ClinicalTermsInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return ClinicalTermsOutput
