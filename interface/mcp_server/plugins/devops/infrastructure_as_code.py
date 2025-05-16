from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class IaCInput(PluginInput):
    template: str = Field(..., description="IaC template content")
    parameters: Dict[str, Any] = Field({}, description="Template parameters")
    provider: str = Field("aws", description="Cloud provider")

class IaCOutput(PluginOutput):
    resources_created: List[str] = Field(..., description="List of created resources")
    outputs: Dict[str, Any] = Field({}, description="Stack outputs")
    warnings: List[str] = Field([], description="Deployment warnings")

class InfrastructureAsCodePlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.iac_engine = None
    
    async def initialize(self):
        from devops_tools import IaCEngine
        self.iac_engine = IaCEngine(self.config)
        await self.iac_engine.initialize()
        self.initialized = True
    
    async def execute(self, input_data: IaCInput) -> IaCOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        result = await self.iac_engine.deploy(
            input_data.template,
            input_data.parameters,
            input_data.provider
        )
        
        return IaCOutput(
            resources_created=result.resources,
            outputs=result.outputs,
            warnings=result.warnings
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Infrastructure as Code Plugin",
            "version": "1.0.0",
            "description": "Manages cloud infrastructure using Infrastructure as Code"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return IaCInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return IaCOutput
