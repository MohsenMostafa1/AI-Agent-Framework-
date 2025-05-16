from core.agent import BaseAgent
from core.tools import SocialMediaTool, AnalyticsTool
from core.memory import CampaignMemory
from typing import List, Dict
import pandas as pd

class MarketingAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "marketing"
        self.tools.register_tool("analyze_campaign", self.analyze_campaign)
        self.tools.register_tool("generate_content", self.generate_content)
        self.tools.register_tool("optimize_ad_copy", self.optimize_ad_copy)
        self.memory = CampaignMemory()
        
    def analyze_campaign(self, metrics: Dict) -> Dict:
        """Analyze marketing campaign performance"""
        return self.llm.generate(
            prompt=f"Analyze these campaign metrics: {metrics}",
            response_format={"type": "json_object"}
        )
    
    def generate_content(self, brief: str, platform: str = "generic") -> Dict:
        """Generate marketing content for specific platforms"""
        return {
            "headline": self.llm.generate(prompt=f"Write {platform} headline about: {brief}"),
            "body": self.llm.generate(prompt=f"Write {platform} content about: {brief}"),
            "cta": self.llm.generate(prompt=f"Write call-to-action for: {brief}")
        }
    
    def optimize_ad_copy(self, ad_text: str, target_audience: str) -> str:
        """Optimize ad copy for better performance"""
        return self.llm.generate(
            prompt=f"Optimize this ad for {target_audience}: {ad_text}",
            temperature=0.7
        )
    
    def create_content_calendar(self, themes: List[str], duration: int = 30) -> pd.DataFrame:
        """Generate a content calendar"""
        return self.llm.generate(
            prompt=f"Create {duration}-day content calendar for themes: {themes}",
            response_format={"type": "json_object"}
        )
