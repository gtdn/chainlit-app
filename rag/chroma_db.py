import logging
import chromadb


logger = logging.getLogger(__name__)


class ChromaDB:
    def __init__(
        self,
        collection_name,
        host,
        port,
    ):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_settings(cls, settings):
        return cls(
            collection_name=settings.chroma_collection,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        
    def _get_or_create_collection(self, collection_name):
        return self.client.get_or_create_collection(name=collection_name)

    def view_collection(self, collection_name):
        collection = self.client.get_collection(collection_name)
        print(collection.peek())
        self.collection = collection

    def add_documents(self, documents):
        if not self.collection:
            raise ValueError("Collection not initialized. Call connect() first.")

        self.collection.add(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[{"source": doc["source"]} for doc in documents],
            embeddings=[doc["vector"] for doc in documents],
        )

    def search(self, embedding, n_results: int = 5):

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "text": text,
                "source": (meta or {}).get("source", "inconnue"),
                "distance": distance,
            }
            for text, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
