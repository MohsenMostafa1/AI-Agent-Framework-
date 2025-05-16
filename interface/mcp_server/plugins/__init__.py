from typing import Dict, Type, Any
from importlib import import_module
from pathlib import Path
import yaml

from .base_plugin import BasePlugin

class PluginManager:
    def __init__(self, config_path: str = "configs/plugin_configs.yaml"):
        self.config = self._load_config(config_path)
        self.plugins: Dict[str, BasePlugin] = {}
        self._discover_plugins()
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _discover_plugins(self):
        plugins_dir = Path(__file__).parent
        for plugin_dir in plugins_dir.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith('_'):
                try:
                    module = import_module(f"plugins.{plugin_dir.name}")
                    for attr in dir(module):
                        if attr.endswith('Plugin') and attr != 'BasePlugin':
                            plugin_class = getattr(module, attr)
                            self.register_plugin(plugin_class.__name__, plugin_class)
                except ImportError as e:
                    print(f"Failed to load plugin {plugin_dir.name}: {e}")
    
    def register_plugin(self, name: str, plugin_class: Type[BasePlugin]):
        if name in self.plugins:
            raise ValueError(f"Plugin {name} already registered")
        self.plugins[name] = plugin_class
    
    async def get_plugin(self, name: str) -> BasePlugin:
        if name not in self.plugins:
            raise ValueError(f"Plugin {name} not found")
        
        if name not in self._initialized_plugins:
            plugin_config = self.config.get(name, {})
            plugin = self.plugins[name](plugin_config)
            await plugin.initialize()
            self._initialized_plugins[name] = plugin
        
        return self._initialized_plugins[name]
    
    async def execute_plugin(self, name: str, input_data: Any) -> Any:
        plugin = await self.get_plugin(name)
        return await plugin.execute(input_data)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "metadata": plugin_class.metadata,
                "input_schema": plugin_class.input_schema.schema(),
                "output_schema": plugin_class.output_schema.schema()
            }
            for name, plugin_class in self.plugins.items()
        }
