from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class DeploymentInput(PluginInput):
    application: str = Field(..., description="Application to deploy")
    environment: str = Field(..., description="Target environment")
    version: str = Field(..., description="Version to deploy")
    config_overrides: Dict[str, Any] = Field({}, description="Configuration overrides")

class DeploymentOutput(PluginOutput):
    success: bool = Field(..., description="Deployment status")
    logs: List[str] = Field([], description="Deployment logs")
    endpoints: List[str] = Field([], description="Access endpoints")
    warnings: List[str] = Field([], description="Deployment warnings")

class DeploymentAutomationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.deployer = None
    
    async def initialize(self):
        from devops_tools import DeploymentManager
        self.deployer = DeploymentManager(self.config)
        await self.deployer.connect()
        self.initialized = True
    
    async def execute(self, input_data: DeploymentInput) -> DeploymentOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        result = await self.deployer.deploy(
            input_data.application,
            input_data.environment,
            input_data.version,
            input_data.config_overrides
        )
        
        return DeploymentOutput(
            success=result.success,
            logs=result.logs,
            endpoints=result.endpoints,
            warnings=result.warnings
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Deployment Automation Plugin",
            "version": "1.0.0",
            "description": "Automates application deployments to various environments"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return DeploymentInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return DeploymentOutput
