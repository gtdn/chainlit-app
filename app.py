from config import get_settings
from llm.client import OpenAIClient
from llm.embeddings import EmbeddingsClient
from llm.router import RouterLLM
from ui.chat_ui import ChatUI
from tools.base import ToolRegistry
from tools.rag import RagTool
from rag.chroma_db import ChromaDB


settings = get_settings()

chroma_db = ChromaDB.from_settings(settings)
embeddings_client = EmbeddingsClient(settings.embedding_model)

tools = ToolRegistry(
    [
        RagTool(
            embedder=embeddings_client,
            vector_store=chroma_db,
            n_results=settings.rag_n_results,
            max_distance=settings.rag_max_distance,
        ),
    ]
)

openai_client = OpenAIClient(settings=settings, tools=tools)

router = RouterLLM(settings=settings, openai_client=openai_client)


chat_ui = ChatUI(openai_client, router)
