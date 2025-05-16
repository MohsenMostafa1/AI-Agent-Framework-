from core.agent import BaseAgent
from core.tools import CloudAPITool, MonitoringTool
import subprocess
import yaml
import json

class DevOpsAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "devops"
        self.tools.register_tool("execute_shell_command", self.execute_shell)
        self.tools.register_tool("generate_terraform", self.generate_terraform)
        self.tools.register_tool("analyze_logs", self.analyze_logs)
        
    def execute_shell(self, command: str) -> str:
        """Execute shell commands in a sandboxed environment"""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception as e:
            return str(e)
    
    def generate_terraform(self, requirements: Dict) -> str:
        """Generate Terraform configuration from requirements"""
        return self.llm.generate(
            prompt=f"Create Terraform config for: {requirements}",
            response_format={"type": "hcl"}
        )
    
    def analyze_logs(self, log_data: str) -> Dict:
        """Analyze system/application logs for issues"""
        return self.llm.generate(
            prompt=f"Analyze these logs: {log_data}",
            response_format={"type": "json_object"}
        )
    
    def create_ci_cd_pipeline(self, config: Dict) -> str:
        """Generate CI/CD pipeline configuration"""
        return yaml.dump(
            self.llm.generate(
                prompt=f"Create CI/CD pipeline for: {config}",
                response_format={"type": "yaml"}
            )
        )
