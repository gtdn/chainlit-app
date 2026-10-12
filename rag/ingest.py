import sys
import time
import argparse
import os
import logging

from pathlib import Path
from config import get_settings
from llm.embeddings import EmbeddingsClient
from rag.ocr_client import OCRClient
from rag.chroma_db import ChromaDB


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Ingère des documents dans ChromaDB")
    parser.add_argument("folder", type=Path, help="Dossier contenant les documents")
    return parser.parse_args()


def main(document_path):
    settings = get_settings()
    chroma_db = ChromaDB.from_settings(settings)
    embedding_client = EmbeddingsClient(settings)

    ocr = OCRClient(settings, embedding_client)
    for file_path in os.listdir(document_path):
        logging.info(f"\n--- Traitement du fichier : {file_path} ---")
        start = time.perf_counter()
        chunks = ocr.process_document(os.path.join(document_path, file_path))
        elapsed = time.perf_counter() - start

        logging.info(f"\n--- Résultat ---")
        logging.info(f"Fichier       : {file_path}")
        logging.info(f"Chunks créés  : {len(chunks)}")
        logging.info(f"Temps total   : {elapsed:.2f}s")

        logging.info(f"Ingestion du {file_path} dans la base chroma DB")
        chroma_db.add_documents(chunks)

    chroma_db.view_collection("rag_collection")


if __name__ == "__main__":
    args = parse_args()
    if not args.folder.is_dir():
        sys.exit(f"Erreur : dossier introuvable -> {args.folder}")
    main(args.folder)
