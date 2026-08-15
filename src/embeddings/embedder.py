"""
embedder.py

Converts transcript chunks into dense vector embeddings using a
Sentence Transformers model. Kept modular so the embedding model
can be swapped later without touching the rest of the pipeline.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EmbeddedChunk:
    chunk_id: str
    text: str
    start: float
    end: float
    speaker: Optional[str]
    meeting_id: Optional[str]
    embedding: List[float]


class Embedder:
    """
    Wraps a SentenceTransformer model to convert text chunks into
    dense vector embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def embed_chunks(self, chunks: List[dict]) -> List[EmbeddedChunk]:
        """
        Args:
            chunks: list of chunk dicts (chunk_id, text, start, end, speaker, meeting_id)

        Returns:
            List of EmbeddedChunk objects with embeddings attached.
        """
        if not chunks:
            logger.warning("No chunks provided to embed.")
            return []

        texts = [c["text"] for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunk(s)...")
        vectors = self.model.encode(texts, show_progress_bar=True)

        embedded: List[EmbeddedChunk] = []
        for chunk, vector in zip(chunks, vectors):
            embedded.append(
                EmbeddedChunk(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    start=chunk["start"],
                    end=chunk["end"],
                    speaker=chunk.get("speaker"),
                    meeting_id=chunk.get("meeting_id"),
                    embedding=vector.tolist(),
                )
            )

        logger.info(f"Generated {len(embedded)} embedding(s). Dimension: {len(embedded[0].embedding)}")
        return embedded

    @staticmethod
    def load_chunks(json_path: str) -> List[dict]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks JSON not found: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_embedded_chunks(embedded_chunks: List[EmbeddedChunk], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in embedded_chunks]
        out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved {len(embedded_chunks)} embedded chunks to {out}")


if __name__ == "__main__":
    # Manual test runner.
    # Usage: python embedder.py data/processed/sample_chunks.json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <chunks_json_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    embedder = Embedder(model_name=model_name)
    chunks = embedder.load_chunks(input_path)
    embedded = embedder.embed_chunks(chunks)

    print(f"\nFirst chunk embedding preview (first 10 dims):")
    print(embedded[0].embedding[:10])
    print(f"\nTotal embedding dimension: {len(embedded[0].embedding)}")

    stem = Path(input_path).stem.replace("_chunks", "")
    embedder.save_embedded_chunks(embedded, f"data/processed/{stem}_embedded.json")
