import os
import logging
import uuid
from paddleocr import PaddleOCR


logger = logging.getLogger(__name__)
logging.getLogger("ppocr").setLevel(logging.ERROR)


class OCRClient:
    def __init__(self, embedding_client):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="fr")
        self.embedding_client = embedding_client
        self.embedding_model = "nomic-embed-text:137m-v1.5-fp16"
        self.chunk_size = 1000
        self.chunk_overlap = 100

    def _extract_text(self, file_path):
        print(f"Extraction du texte du document : {file_path}")
        result = self.ocr.ocr(file_path)

        if not result or result[0] is None:
            return ""

        lines = []
        for page in result:
            for detection in page:
                text = detection[1][0]
                lines.append(text)

        print(f"Texte extrait : {len(lines)} lignes")

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
        response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def process_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")

        text = self._extract_text(file_path)

        if not text:
            print(f"Aucun texte extrait pour {file_path}")
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
