import pytest
from core.agent import Agent
from core.tools import ToolRegistry, WebSearchTool
from unittest.mock import MagicMock, patch

class TestToolCalling:
    @pytest.fixture
    def tool_registry(self):
        registry = ToolRegistry()
        registry.register_tool(WebSearchTool())
        return registry

    @pytest.fixture
    def agent(self, tool_registry):
        return Agent(tool_registry=tool_registry)

    def test_tool_registration(self, tool_registry):
        assert 'web_search' in tool_registry.tools, "Web search tool should be registered"
        assert tool_registry.tools['web_search'].description, "Tool should have description"

    def test_tool_descriptions_for_llm(self, tool_registry):
        descriptions = tool_registry.get_tool_descriptions()
        assert isinstance(descriptions, list), "Should return list of descriptions"
        assert any('web_search' in desc for desc in descriptions), "Should include web search"

    @patch('core.tools.WebSearchTool.execute')
    def test_agent_tool_execution(self, mock_execute, agent):
        mock_execute.return_value = "Search results for weather"
        
        # Simulate LLM choosing to use the web search tool
        llm_response = {
            "function_call": {
                "name": "web_search",
                "arguments": '{"query":"current weather in London"}'
            }
        }
        
        result = agent.handle_llm_response(llm_response)
        assert result == "Search results for weather", "Should return tool result"
        mock_execute.assert_called_once_with(query="current weather in London")

    def test_parallel_tool_execution(self, agent):
        # Mock tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.execute.return_value = "result1"
        
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.execute.return_value = "result2"
        
        agent.tool_registry.register_tool(tool1)
        agent.tool_registry.register_tool(tool2)
        
        # Simulate parallel tool calls
        llm_response = {
            "function_calls": [
                {"name": "tool1", "arguments": "{}"},
                {"name": "tool2", "arguments": "{}"}
            ]
        }
        
        results = agent.handle_llm_response(llm_response)
        assert len(results) == 2, "Should return results from both tools"
        assert results[0] == "result1", "First result should match tool1 output"
        assert results[1] == "result2", "Second result should match tool2 output"
