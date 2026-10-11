import sys
import time
from openai import OpenAI
from ocr_client import OCRClient
from dotenv import load_dotenv
import os
from chroma_db import ChromaDB

load_dotenv()


def main(document_path):
    chroma_db = ChromaDB(
        persist_directory="chroma_db", collection_name="rag_collection"
    )

    embedding_client = OpenAI(
        base_url=os.getenv("LITE_LLM_BASE"),
        api_key=os.getenv("LITE_LLM_KEY"),
    )

    ocr = OCRClient(embedding_client)
    for file_path in os.listdir(document_path):
        print(f"\n--- Traitement du fichier : {file_path} ---")
        start = time.perf_counter()
        chunks = ocr.process_document(document_path + file_path)
        elapsed = time.perf_counter() - start

        print(f"\n--- Résultat ---")
        print(f"Fichier       : {file_path}")
        print(f"Chunks créés  : {len(chunks)}")
        print(f"Temps total   : {elapsed:.2f}s")

        # for i, chunk in enumerate(chunks):
        # print(f"\n[Chunk {i}] ({len(chunk['text'])} caractères)")
        # print(f"Texte   : {chunk['text'][:200]}{'...' if len(chunk['text']) > 200 else ''}")
        # print(f"Vecteur : dimension {len(chunk['vector'])}, premières valeurs {chunk['vector'][:5]}")
        # print(f"Source  : {chunk['Source']:.4f}")

        print(f"Ingestion des {file_path} dans la base chroma DB")
        chroma_db.add_documents(chunks)

    chroma_db.view_collection("rag_collection")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <chemin_du_fichier>")
        sys.exit(1)

    main(sys.argv[1])
