from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    def __init__(self, settings=None, tools=None):
        self.client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )
        self.async_client = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key
        )
        self.prompt_system = self._get_prompt_system()
        self.tools = tools

    def _get_prompt_system(self):
        return """
            You are a helpful assistant specialist of giving advices about camping gears, for the compagny Contoso : backpacks, tents, accesoiries
            So answer witht the speficics products we sell, use your tool for that. Use many emojis to be friendly.
        """

    def completion(
        self,
        prompt_system=None,
        prompt=None,
        model="llama3.2:latest",
        response_format=None,
    ):
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

    async def completion_stream(self, messages, model="llama3.2:latest"):
        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            yield chunk

    def completion_tools(self, messages, model="llama3.2:latest"):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=self.tools.schemas,
        )
        return response
