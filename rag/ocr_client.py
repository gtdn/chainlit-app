import os
import logging
import uuid
from paddleocr import PaddleOCR


logger = logging.getLogger(__name__)
logging.getLogger("ppocr").setLevel(logging.ERROR)


class OCRClient:
    def __init__(self, settings, embedding_client):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="fr")
        self.embedding_client = embedding_client
        self.embedding_model = settings.embedding_model
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def _extract_text(self, file_path):
        logging.info(f"Extraction du texte du document : {file_path}")
        result = self.ocr.ocr(file_path)
        if not result or result[0] is None:
            return ""

        lines = []
        for page in result:
            for detection in page:
                text = detection[1][0]
                lines.append(text)

        logging.info(f"Texte extrait : {len(lines)} lignes")

        return "\n".join(lines)

    def _chunk_text(self, text):
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        return chunks

    def _embed(self, text):
        return self.embedding_client.get_embeddings(
            text=text,
        )

    def process_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        text = self._extract_text(file_path)

        if not text:
            logging.warning(f"Aucun texte extrait pour {file_path}")
            return []

        chunks = self._chunk_text(text)

        results = []
        for chunk in chunks:
            vector = self._embed(chunk)
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "vector": vector,
                    "source": os.path.basename(file_path),
                }
            )

        return results
