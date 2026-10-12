import json
import logging


class RouterLLM:
    def __init__(self, settings, openai_client):
        self.openai_client = openai_client
        self.default_model = settings.default_model
        self.routing_model = settings.routing_model
        self.catalog_models = {
            "groq/qwen/qwen3.8-27b": "Big modèle for complex tasks, tools callings, code. Expensive, use only for complex tasks.",
            "groq/openai/gpt-oss-20b": "Medium modele for medium complex tasks, tools callings, generate code. Moderate cost, suitable for most tasks.",
            "groq/openai/gpt-oss-20b": "Small modele for just for simple answers and super simple tasks. Cheap, use for simple tasks.",
        }
        self.prompt_system = self._get_prompt_system()

    def _get_prompt_system(self):
        return f"""
            You are a router for LLM requests. You will receive a request and determine which LLM provider to use based on the request content. 
            You will then forward the request to the appropriate LLM provider and return the response.
            Choose the LLM according to the complexity, to not overload the big models with simple requests.

            Tools availables : RAG for informations about campings gears (tents, backpacks, accesories)
            
            LLM availables:
            {self.catalog_models}

            Never answer the content of the request. 
            Only respond with a JSON strictly in the form:
            {{"model": "<exact name of the chosen model>"}}
        """

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
