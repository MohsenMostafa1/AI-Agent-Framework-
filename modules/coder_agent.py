from core.agent import BaseAgent
from core.tools import CodeSearchTool, DocumentationTool
from core.memory import CodeMemory
from typing import List, Dict
import ast

class CoderAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "software_development"
        self.tools.register_tool("generate_code", self.generate_code)
        self.tools.register_tool("debug_code", self.debug_code)
        self.tools.register_tool("refactor_code", self.refactor_code)
        self.memory = CodeMemory()
        
    def generate_code(self, requirements: str, language: str = "python") -> str:
        """Generate code from requirements"""
        return self.llm.generate(
            prompt=f"Write {language} code that: {requirements}",
            response_format={"type": "text"}
        )
    
    def debug_code(self, code: str, error: str = None) -> Dict:
        """Debug and fix code issues"""
        if not error:
            error = self.analyze_code_errors(code)
            
        fixed_code = self.llm.generate(
            prompt=f"Fix this code: {code}. Error: {error}",
            response_format={"type": "text"}
        )
        return {
            "fixed_code": fixed_code,
            "explanation": self.explain_fix(code, fixed_code)
        }
    
    def refactor_code(self, code: str, style: str = "pep8") -> str:
        """Refactor code to meet style guidelines"""
        return self.llm.generate(
            prompt=f"Refactor this code to {style} standards: {code}",
            response_format={"type": "text"}
        )
    
    def analyze_code_errors(self, code: str) -> str:
        """Static analysis of code for potential issues"""
        try:
            ast.parse(code)
            return "No syntax errors detected"
        except SyntaxError as e:
            return str(e)
    
    def explain_fix(self, original: str, fixed: str) -> str:
        """Explain the changes made during debugging"""
        return self.llm.generate(
            prompt=f"Explain the difference between: {original} and {fixed}",
            temperature=0.7
        )
