import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Tool(ABC):
    name: str
    description: str
    parameters: dict

    @abstractmethod
    def run(self, **kwargs):
        """
        Run tool with the given arguments.
        """

    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools):
        self._tools = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool):
        if tool.name in self._tools:
            raise ValueError(f"Outil déjà enregistré : {tool.name}")
        self._tools[tool.name] = tool

    @property
    def schemas(self):
        return [tool.schema for tool in self._tools.values()]

    def execute(self, tool_call):
        name = tool_call.function.name
        tool = self._tools.get(name)
        if tool is None:
            return f"Erreur : outil '{name}' inconnu."

        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return f"Erreur : arguments invalides pour '{name}' ({e})."

        try:
            logger.info("Exécution de l'outil %s avec %s", name, args)
            return str(tool.run(**args))
        except Exception as e:
            logger.exception("Échec de l'outil %s", name)
            return f"Erreur pendant l'exécution de '{name}' : {e}"
