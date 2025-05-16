from core.agent import BaseAgent
from core.tools import SECSearchTool, FinancialDataTool
from core.memory import FinancialMemory
from datetime import datetime
import yfinance as yf
import pandas as pd

class FinanceAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.specialization = "finance"
        self.tools.register_tool("get_stock_data", self.get_stock_data)
        self.tools.register_tool("analyze_sec_filings", self.analyze_sec_filings)
        self.tools.register_tool("calculate_financial_ratios", self.calculate_ratios)
        self.memory = FinancialMemory()
        
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Retrieve historical stock data"""
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        self.memory.store_market_data(ticker, hist.to_dict())
        return hist
    
    def analyze_sec_filings(self, company: str, filing_type: str = "10-K") -> str:
        """Analyze SEC filings using RAG"""
        results = self.retriever.search(
            query=f"{company} {filing_type}",
            filters={"source": "sec.gov"}
        )
        analysis = self.llm.generate(
            prompt=f"Analyze this SEC filing: {results}",
            response_format={"type": "json_object"}
        )
        return analysis
    
    def calculate_ratios(self, financials: Dict) -> Dict:
        """Calculate key financial ratios"""
        ratios = {
            "pe_ratio": financials["price"] / financials["eps"],
            "current_ratio": financials["current_assets"] / financials["current_liabilities"],
            "debt_to_equity": financials["total_debt"] / financials["total_equity"]
        }
        return ratios
    
    def generate_investment_thesis(self, ticker: str) -> str:
        """Generate comprehensive investment analysis"""
        data = self.get_stock_data(ticker)
        filings = self.analyze_sec_filings(ticker)
        return self.llm.generate(
            prompt=f"Create investment thesis for {ticker}",
            temperature=0.3
        )
