import logging
from openai import OpenAI, AsyncOpenAI

from llm.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, settings, tools=None):
        self.client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )
        self.async_client = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )
        self.prompt_system = SYSTEM_PROMPT
        self.default_model = settings.default_model
        self.tools = tools

    def completion(
        self,
        prompt_system=None,
        prompt=None,
        model=None,
        response_format=None,
    ):
        if not model:
            model = self.default_model

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": prompt_system or self.prompt_system,
                },
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
        )
        return response.choices[0].message.content

    async def completion_stream(self, messages, model):
        if not model:
            model = self.default_model

        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            yield chunk

    def completion_tools(self, messages, model):
        if not model:
            model = self.default_model

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=self.tools.schemas,
        )
        return response
