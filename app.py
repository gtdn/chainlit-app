from openai_client import OpenAIClient
from routerLLM import RouterLLM
from chat_ui import ChatUI
from tools import Tools
from documents_rag.chroma_db import ChromaDB
from documents_rag.ocr_client import OCRClient


chroma_db = ChromaDB()
tools = Tools(chroma_db=chroma_db)
openai_client = OpenAIClient(tools=tools)
router = RouterLLM()
ocr_client = OCRClient(openai_client)


chat_ui = ChatUI(openai_client, router, ocr_client)
