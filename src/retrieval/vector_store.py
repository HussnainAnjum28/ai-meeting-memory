"""
vector_store.py

Wraps ChromaDB to store embedded transcript chunks and perform
semantic similarity search over them.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStore:
    def __init__(self, persist_directory: str = "data/vector_store", collection_name: str = "meetings"):
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store ready. Collection has {self.collection.count()} existing chunk(s).")

    def add_embedded_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        if not embedded_chunks:
            logger.warning("No embedded chunks provided to add.")
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in embedded_chunks:
            unique_id = f"{chunk.get('meeting_id', 'unknown')}_{chunk['chunk_id']}"
            ids.append(unique_id)
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "speaker": chunk.get("speaker") or "",
                "meeting_id": chunk.get("meeting_id") or "",
            })

        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Added/updated {len(ids)} chunk(s) in the vector store.")

    def query(self, query_embedding: List[float], top_k: int = 5, meeting_id: Optional[str] = None) -> List[Dict[str, Any]]:
        where_filter = {"meeting_id": meeting_id} if meeting_id else None
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_filter)

        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

        logger.info(f"Query returned {len(formatted_results)} result(s).")
        return formatted_results

    def delete_meeting(self, meeting_id: str) -> None:
        self.collection.delete(where={"meeting_id": meeting_id})
        logger.info(f"Deleted all chunks for meeting_id='{meeting_id}'.")

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def load_embedded_chunks(json_path: str) -> List[Dict[str, Any]]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Embedded chunks JSON not found: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <embedded_chunks_json_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    store = VectorStore()
    chunks = store.load_embedded_chunks(input_path)
    store.add_embedded_chunks(chunks)

    print(f"\nTotal chunks in store: {store.count()}")

    test_embedding = chunks[0]["embedding"]
    results = store.query(test_embedding, top_k=3)

    print("\n--- Sample Query Results (using first chunk as query) ---")
    for r in results:
        print(f"\nID: {r['id']} | distance: {r['distance']:.4f}")
        print(f"Timestamp: {r['metadata']['start']}-{r['metadata']['end']}")
        print(f"Text: {r['text'][:150]}...")
