from typing import Dict, List
from ..base_plugin import BasePlugin, PluginInput, PluginOutput
from pydantic import BaseModel, Field

class MonitoringInput(PluginInput):
    resources: List[str] = Field(..., description="Resources to monitor")
    metrics: List[str] = Field(..., description="Metrics to collect")
    alert_rules: Dict[str, Any] = Field({}, description="Alert rules configuration")

class MonitoringOutput(PluginOutput):
    dashboard_url: str = Field(..., description="Monitoring dashboard URL")
    active_alerts: List[Dict[str, Any]] = Field([], description="Current active alerts")
    metrics_data: Dict[str, Any] = Field({}, description="Collected metrics data")

class MonitoringIntegrationPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.monitoring_client = None
    
    async def initialize(self):
        from devops_tools import MonitoringClient
        self.monitoring_client = MonitoringClient(self.config)
        await self.monitoring_client.connect()
        self.initialized = True
    
    async def execute(self, input_data: MonitoringInput) -> MonitoringOutput:
        if not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        await self.monitoring_client.setup_monitoring(
            input_data.resources,
            input_data.metrics,
            input_data.alert_rules
        )
        
        dashboard = await self.monitoring_client.get_dashboard()
        alerts = await self.monitoring_client.get_active_alerts()
        metrics = await self.monitoring_client.get_metrics()
        
        return MonitoringOutput(
            dashboard_url=dashboard.url,
            active_alerts=alerts,
            metrics_data=metrics
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Monitoring Integration Plugin",
            "version": "1.0.0",
            "description": "Integrates with monitoring systems to track infrastructure"
        }
    
    @property
    def input_schema(self) -> type[PluginInput]:
        return MonitoringInput
    
    @property
    def output_schema(self) -> type[PluginOutput]:
        return MonitoringOutput
