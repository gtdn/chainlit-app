# tools.py
import json
import os
from openai import OpenAI


class Tools:
    def __init__(
        self,
        chroma_db,
    ):
        self.base_url = os.getenv("LITE_LLM_BASE")
        self.api_key = os.getenv("LITE_LLM_KEY")
        self.openai_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.chroma_db = chroma_db

        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Récupère la météo actuelle pour une ville donnée.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "Le nom de la ville, ex: Paris",
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Évalue une expression mathématique simple.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Expression mathématique, ex: '3 * (2 + 5)'",
                            },
                        },
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_chroma",
                    "description": "Ask the chroma database for information about camping gears",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search in the chroma database",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

        self._registry = {
            "get_weather": self._get_weather,
            "calculate": self._calculate,
            "query_chroma": self._query_chroma,
        }

    def _get_weather(self, city):
        return f"Il fait 22°C et ensoleillé à {city}."

    def _calculate(self, expression):
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Erreur de calcul : {e}"

    def _query_chroma(self, query):
        print(f"Querying ChromaDB with: {query}")
        query_embedding = self.openai_client.embeddings.create(
            model="nomic-embed-text:137m-v1.5-fp16", input=query
        )
        results = self.chroma_db.query(query_embedding.data[0].embedding, n_results=5)
        return results

    def execute(self, tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        func = self._registry.get(name)
        if func is None:
            return f"Erreur : tool '{name}' inconnu."

        return func(**args)

    def embeding(self, message, model="nomic-embed-text:137m-v1.5-fp16"):
        reponse = self.client.embeddings.create(
            model=model,
            input=message,
        )
        return reponse.data[0].embedding
