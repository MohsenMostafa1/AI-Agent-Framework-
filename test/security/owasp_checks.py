import pytest
import json
from core.api import app  # FastAPI app
from fastapi.testclient import TestClient
from core.agent import Agent
from unittest.mock import patch

class TestSecurityOWASP:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_sql_injection_prevention(self, client):
        # Attempt SQL injection via prompt
        response = client.post("/api/process", json={
            "prompt": "'; DROP TABLE users; --"
        })
        
        assert response.status_code == 200, "Should handle malicious input gracefully"
        assert "DROP TABLE" not in response.text, "Should not execute SQL commands"

    def test_xss_protection(self, client):
        # Attempt XSS via prompt
        xss_payload = "<script>alert('XSS')</script>"
        response = client.post("/api/process", json={
            "prompt": f"What is {xss_payload}?"
        })
        
        assert response.status_code == 200
        assert xss_payload not in response.text, "Should sanitize XSS payloads"

    def test_rate_limiting(self, client):
        # Test basic rate limiting
        responses = []
        for _ in range(11):  # Assuming limit is 10/min
            response = client.post("/api/process", json={
                "prompt": "Rate limit test"
            })
            responses.append(response.status_code)
        
        # Last request should be rate limited
        assert 429 in responses, "Should enforce rate limiting"

    def test_sensitive_data_exposure(self, client):
        # Test that error responses don't expose sensitive info
        with patch.object(Agent, 'process', side_effect=Exception("DB password is 12345")):
            response = client.post("/api/process", json={
                "prompt": "Cause error"
            })
            
            assert response.status_code == 500
            assert "12345" not in response.text, "Should not expose sensitive data"

    def test_insecure_deserialization(self, client):
        # Attempt to send malicious payload
        malicious_payload = {
            "__class__": "subprocess.Popen",
            "__init__": {"args": "rm -rf /", "shell": True}
        }
        
        response = client.post("/api/process", 
            content=json.dumps(malicious_payload),
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422], "Should reject malicious payloads"
