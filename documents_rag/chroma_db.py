import chromadb


class ChromaDB:
    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="rag_collection",
        host="localhost",
        port=8000,
    ):
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.client = chromadb.HttpClient(host=self.host, port=self.port)
        self.collection = self.create_collection(collection_name)

    def create_collection(self, collection_name):
        if not self.client.get_collection(collection_name):
            self.client.create_collection(name=collection_name)
            return self.client.get_collection(collection_name)

        return self.client.get_collection(collection_name)

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

    def query(self, query_emb, n_results=5):
        if not self.collection:
            raise ValueError("Collection not initialized. Call connect() first.")

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
        )

        texts_chunks = results["documents"][0]
        return texts_chunks
