from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class EmbeddingsClient:
    def __init__(self, embedding_model):
        self.base_url = os.getenv("LITE_LLM_BASE")
        self.api_key = os.getenv("LITE_LLM_KEY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = embedding_model

    def get_embeddings(self, text):
        response = self.client.embeddings.create(input=text, model=self.model)

        return response.data[0].embedding