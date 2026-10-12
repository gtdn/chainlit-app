import logging

from openai import OpenAI


logger = logging.getLogger(__name__)


class EmbeddingsClient:
    def __init__(self, settings):
        self.client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )
        self.model = settings.embedding_model

    def get_embeddings(self, text):
        response = self.client.embeddings.create(input=text, model=self.model)

        return response.data[0].embedding
