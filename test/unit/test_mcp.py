import pytest
from core.mcp_integration import MCPClient, MCPServer
from interface.mcp_server.plugins.base_plugin import BasePlugin
from unittest.mock import MagicMock

class TestMCPIntegration:
    @pytest.fixture
    def mock_plugin(self):
        plugin = MagicMock(spec=BasePlugin)
        plugin.name = "test_plugin"
        plugin.version = "1.0"
        plugin.process.return_value = {"result": "processed_data"}
        return plugin

    def test_mcp_protocol_handshake(self):
        client = MCPClient()
        server = MCPServer()
        
        # Simulate handshake
        handshake_response = server.handle_request({
            'protocol': 'MCP',
            'version': '1.0',
            'action': 'handshake'
        })
        
        assert handshake_response['status'] == 'success', "Handshake should succeed"
        assert 'supported_versions' in handshake_response, "Response should include versions"

    def test_plugin_registration(self, mock_plugin):
        server = MCPServer()
        server.register_plugin(mock_plugin)
        
        assert mock_plugin.name in server.plugins, "Plugin should be registered"
        assert server.plugins[mock_plugin.name] == mock_plugin, "Plugin instance should match"

    def test_plugin_execution(self, mock_plugin):
        server = MCPServer()
        server.register_plugin(mock_plugin)
        
        response = server.handle_request({
            'protocol': 'MCP',
            'version': '1.0',
            'action': 'execute',
            'plugin': 'test_plugin',
            'data': {'input': 'test'}
        })
        
        assert response['status'] == 'success', "Execution should succeed"
        assert response['data']['result'] == 'processed_data', "Should return plugin output"
        mock_plugin.process.assert_called_once_with({'input': 'test'})

    def test_mcp_client_integration(self, mock_plugin):
        server = MCPServer()
        server.register_plugin(mock_plugin)
        client = MCPClient(server_url="http://localhost:8000")
        
        # Mock the actual HTTP request
        client._send_request = MagicMock(return_value={
            'status': 'success',
            'data': {'result': 'processed_data'}
        })
        
        response = client.execute_plugin(
            plugin_name="test_plugin",
            data={'input': 'test'}
        )
        
        assert response['status'] == 'success', "Client should receive success response"
        assert response['data']['result'] == 'processed_data', "Client should get processed data"
