from typing import List, Dict, Any
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class ClauseGeneratorInput(PluginInput):
    clause_type: str = Field(..., description="Type of clause to generate")
    parameters: Dict[str, Any] = Field({}, description="Parameters for clause generation")
    style: str = Field("formal", description="Language style for the clause")

class ClauseGeneratorOutput(PluginOutput):
    generated_clause: str = Field(..., description="Generated legal clause")
    alternatives: List[str] = Field([], description="Alternative formulations")
    warnings: List[str] = Field([], description="Any legal warnings")

class ClauseGeneratorPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.template_db = None
        self.style_models = None
    
    async def initialize(self):
        from legal_models import load_clause_templates, load_style_models
        self.template_db = await load_clause_templates()
        self.style_models = await load_style_models()
        self.initialized = True
    
    async def execute(self, input_data: ClauseGeneratorInput) -> ClauseGeneratorOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        template = self.template_db.get_template(input_data.clause_type)
        if not template:
            raise ValueError(f"Unknown clause type: {input_data.clause_type}")
        
        clause = template.generate(input_data.parameters)
        styled_clause = self.style_models.apply_style(clause, input_data.style)
        alternatives = self.style_models.generate_alternatives(clause, 3)
        
        return ClauseGeneratorOutput(
            generated_clause=styled_clause,
            alternatives=alternatives,
            warnings=template.get_warnings(input_data.parameters)
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Clause Generator Plugin",
            "version": "1.0.0",
            "description": "Generates legal clauses based on templates and parameters"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return ClauseGeneratorInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return ClauseGeneratorOutput
