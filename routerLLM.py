from openai import OpenAI, AsyncOpenAI
import os


class RouterLLM:
    def __init__(self):
        self.base_url = os.getenv("LITE_LLM_BASE")
        self.api_key = os.getenv("LITE_LLM_KEY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.default_model = "groq/qwen/qwen3.8-27b"
        self.routing_model = "groq/openai/gpt-oss-20b"
        self.catalog_models = {
            "groq/qwen/qwen3.8-27b": "Big modèle for complex tasks, tools callings, code. Expensive, use only for complex tasks.",
            "groq/openai/gpt-oss-20b": "Medium modèle for medium complex tasks, tools callings, code. Moderate cost, suitable for most tasks.",
            "llama3.2:latest": "Small modèle for simple tasks, text generation, summarization, etc. Low cost, fast response.",
        }
        self.prompt_system = self._get_prompt_system()

    def _get_prompt_system(self):
        return f"""
            You are a router for LLM requests. You will receive a request and determine which LLM provider to use based on the request content. 
            You will then forward the request to the appropriate LLM provider and return the response.
            Choose the LLM according to the complexity, to not overload the big models with simple requests.

            LLM availables:
            {self.catalog_models}

            Ne réponds JAMAIS au contenu de la requête. Réponds uniquement
            avec un JSON strictement de la forme :
            {{"model": "<nom exact du modèle choisi>"}}
        """

    def _completion(self, prompt):
        response = self.client.chat.completions.create(
            model=self.routing_model,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_system,
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    async def route(self, prompt):
        import json

        try:
            raw = await self._completion(prompt)
            data = json.loads(raw)
            model = data.get("model")
            print(f"Routing model: {model}")
            return model if model in self.catalog_models else self.default_model
        except Exception:
            return self.default_model
