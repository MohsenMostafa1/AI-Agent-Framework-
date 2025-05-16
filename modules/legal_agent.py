from core.agent import BaseAgent
from core.retriever import LegalRetriever
from core.tools import LegalResearchTool
from core.memory import CaseMemory
from typing import List, Dict

class LegalAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "legal"
        self.retriever = LegalRetriever(
            index_name="legal_precedents",
            embedding_model="lawbert"
        )
        self.tools.register_tool("research_case_law", self.research_case_law)
        self.tools.register_tool("draft_legal_document", self.draft_legal_document)
        self.tools.register_tool("analyze_contract", self.analyze_contract)
        self.memory = CaseMemory()
        
    def research_case_law(self, query: str, jurisdiction: str = "US") -> List[Dict]:
        """Search relevant case law"""
        results = self.retriever.search(
            query=query,
            filters={"jurisdiction": jurisdiction}
        )
        return results
    
    def draft_legal_document(self, doc_type: str, parameters: Dict) -> str:
        """Draft legal documents with proper formatting"""
        template = self.retriever.get_document_template(doc_type)
        return self.llm.generate(
            prompt=f"Fill this legal template: {template} with {parameters}",
            temperature=0.1  # Low creativity for legal precision
        )
    
    def analyze_contract(self, contract_text: str) -> Dict:
        """Analyze contract terms and identify potential issues"""
        analysis = self.llm.generate(
            prompt=f"Analyze this contract: {contract_text}",
            response_format={"type": "json_object"}
        )
        return {
            "analysis": analysis,
            "red_flags": self.identify_red_flags(contract_text)
        }
    
    def identify_red_flags(self, text: str) -> List[str]:
        """Identify potential legal red flags"""
        return self.llm.generate(
            prompt="Identify legal red flags in this text",
            response_format={"type": "json_object"}
        )
