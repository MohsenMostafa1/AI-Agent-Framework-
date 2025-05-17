### TESTING.md (Detailed Contents)
```markdown
# Testing Framework

## Unit Test Structure
```python
# tests/unit/test_mcp.py
def test_plugin_loading():
    loader = PluginManager()
    loader.load("legal")
    assert "analyze_contract" in loader.registered_actions
