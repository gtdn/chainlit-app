from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    def __init__(self, tools=None):
        self.base_url = os.getenv("LITE_LLM_BASE")
        self.api_key = os.getenv("LITE_LLM_KEY")
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self.prompt_system = self._get_prompt_system()
        self.tools = tools

    def _get_prompt_system(self):
        return """
            You are a helpful assistant specialist of giving advices about camping gears, for the compagny Contoso : backpacks, tents, accesoiries
            So answer witht the speficics products we sell, in your rag base.
        """

    def completion(self, prompt, model="llama3.2:latest"):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt_system,
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    async def completion_stream(self, messages, model="llama3.2:latest"):
        stream = await self.client.chat.completions.create(
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
