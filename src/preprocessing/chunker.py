"""
chunker.py

Groups cleaned transcript segments into overlapping text chunks
suitable for embedding and retrieval.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    meeting_id: Optional[str] = None


class TranscriptChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_segments(self, segments: List[dict], meeting_id: Optional[str] = None) -> List[Chunk]:
        if not segments:
            logger.warning("No segments provided to chunk.")
            return []

        chunks: List[Chunk] = []
        current_texts: List[str] = []
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        current_speakers = set()
        current_len = 0

        def flush_chunk():
            nonlocal current_texts, current_start, current_end, current_speakers, current_len
            if not current_texts:
                return
            merged_text = " ".join(current_texts).strip()
            speaker = next(iter(current_speakers)) if len(current_speakers) == 1 else None
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{len(chunks):04d}",
                    text=merged_text,
                    start=current_start,
                    end=current_end,
                    speaker=speaker,
                    meeting_id=meeting_id,
                )
            )

        for seg in segments:
            seg_text = seg["text"].strip()
            if not seg_text:
                continue

            if current_len + len(seg_text) > self.chunk_size and current_texts:
                flush_chunk()
                overlap_text = chunks[-1].text[-self.chunk_overlap:] if chunks else ""
                current_texts = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
                current_start = seg["start"]
                current_speakers = set()

            if current_start is None:
                current_start = seg["start"]

            current_texts.append(seg_text)
            current_len += len(seg_text) + 1
            current_end = seg["end"]
            if seg.get("speaker"):
                current_speakers.add(seg["speaker"])

        flush_chunk()

        logger.info(f"Created {len(chunks)} chunk(s) from {len(segments)} segment(s).")
        return chunks

    @staticmethod
    def load_cleaned_segments(json_path: str) -> List[dict]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Cleaned transcript JSON not found: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_chunks(chunks: List[Chunk], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in chunks]
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved {len(chunks)} chunks to {out}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <cleaned_transcript_json_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))

    chunker = TranscriptChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    segments = chunker.load_cleaned_segments(input_path)

    meeting_id = Path(input_path).stem.replace("_cleaned", "")
    chunks = chunker.chunk_segments(segments, meeting_id=meeting_id)

    print("\n--- Chunks ---")
    for c in chunks:
        print(f"\n[{c.chunk_id}] ({c.start:.2f}-{c.end:.2f}) speaker={c.speaker}")
        print(c.text)

    stem = Path(input_path).stem.replace("_cleaned", "")
    chunker.save_chunks(chunks, f"data/processed/{stem}_chunks.json")
