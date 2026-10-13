import json
import logging
from llm.prompt import build_router_prompt


class RouterLLM:
    def __init__(self, settings, openai_client):
        self.openai_client = openai_client
        self.default_model = settings.default_model
        self.routing_model = settings.routing_model

        self.big = settings.default_model
        self.medium = settings.routing_model
        self.small = settings.small_model

        self.catalog_models = {
            self.big: "Big modèle for complex tasks, tools callings, code. Expensive, use only for complex tasks.",
            self.medium: "Medium modele for medium complex tasks, tools callings, generate code. Moderate cost, suitable for most tasks.",
            self.small: "Small modele for just for simple answers and super simple tasks. Cheap, use for simple tasks.",
        }
        self.prompt_system = build_router_prompt(
            catalog=self.catalog_models,
            big=self.big,
            medium=self.medium,
            small=self.small,
        )

    def _completion(self, prompt):
        return self.openai_client.completion(
            prompt_system=self.prompt_system,
            model=self.routing_model,
            prompt=prompt,
            response_format={"type": "json_object"},
        )

    def route(self, prompt):
        try:
            model = self._completion(prompt)
            data = json.loads(model)
            model = data.get("model", self.default_model)
            logging.info(f"Routing model: {model}")
            logging.info(f"Available models: {model in self.catalog_models}")
            return model if model in self.catalog_models else self.default_model
        except Exception:
            return self.default_model
