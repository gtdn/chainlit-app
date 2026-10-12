from config import get_settings
from llm.client import OpenAIClient
from llm.embeddings import EmbeddingsClient
from llm.router import RouterLLM
from ui.chat_ui import ChatUI
from tools.base import ToolRegistry
from tools.rag import RagTool
from rag.chroma_db import ChromaDB
import logging


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.info("Initialisation de l'application...")

logger.info("Chargement des paramètres...")
settings = get_settings()
logger.info("Paramètres chargés")

logger.info("Initialisation de la base de données ChromaDB...")
chroma_db = ChromaDB.from_settings(settings)
logger.info("Base de données ChromaDB initialisée")

logger.info("Initialisation du client d'embeddings...")
embeddings_client = EmbeddingsClient(settings)
logger.info("Client d'embeddings initialisé")

logger.info("Initialisation des Tools...")
tools = ToolRegistry(
    [
        RagTool(
            embeddings_client=embeddings_client,
            vector_store=chroma_db,
            n_results=settings.rag_n_results,
            max_distance=settings.rag_max_distance,
        ),
    ]
)
logger.info("Tools initialisés")

logger.info("Initialisation du client OpenAI...")
openai_client = OpenAIClient(settings=settings, tools=tools)
logger.info("Client OpenAI initialisé")

logger.info("Initialisation du RouterLLM...")
router = RouterLLM(settings=settings, openai_client=openai_client)
logger.info("RouterLLM initialisé")

logger.info("Initialisation de l'interface utilisateur ChatUI...")
chat_ui = ChatUI(openai_client, router)
logger.info("Interface utilisateur ChatUI initialisée")
