from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

# MovieVectorStore is a DataBase adapter that make the system components modular.
# It provide us with a stable API for our application to interact with any databse that supports vectors.
# This impementation is basic and cover basic functionalities


class MovieVectorStore:

    def __init__(self, persist_dir: str, collection_name: str):
        # PersistentClient writes to disk; survives process restarts.
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self._collection = None  # lazy-initialized

    #
    # setters interface

    @property
    def collection(self):

        #lazy load to avoid database_is_connected for each querry

        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # use cosine similarity to check the embedding distance from each other.
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset(self) -> None:
        # Drop and recreate the collection. For clean rebuilds 
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection didn't exist, no problem
        self._collection = None

    def upsert_batched(self, ids, embeddings, metadatas, documents, chunk: int = 1000):
        # using the upsert function to directly update and insert when needed.
        # avoid checking directly update or create a new vector
        for i in range(0, len(ids), chunk):
            self.collection.upsert(
                ids=ids[i:i + chunk],
                embeddings=embeddings[i:i + chunk],
                metadatas=metadatas[i:i + chunk],
                documents=documents[i:i + chunk],
            )

    #
    # getters interface


    def get_embedding(self, movie_id: str) -> Optional[list]:
        # get the stored embedding for given movie id
        r = self.collection.get(ids=[movie_id], include=["embeddings"])
        # return the embedding if found otherwise return none if the results is empty.
        return r["embeddings"][0] if r["ids"] else None

    def get_metadata(self, movie_id: str) -> Optional[dict]:
        # get the metadata for the given movie id
        r = self.collection.get(ids=[movie_id], include=["metadatas"])
        return r["metadatas"][0] if r["ids"] else None

    def query_by_vector(self, vector, n_results: int) -> dict:
        # find and get up to n_results vector that nearest to our vector using distance defined in constructor defaults to cosine
        return self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()



# Factory: lazy instantiation + caching per version.S

_stores: dict[str, MovieVectorStore] = {}


def get_store(version: str) -> MovieVectorStore:
    if version not in _stores:
        _stores[version] = MovieVectorStore(
            persist_dir=settings.chroma_db_dir,
            collection_name=settings.collection_name(version),
        )
    return _stores[version]