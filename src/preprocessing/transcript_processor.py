"""
transcript_processor.py

Cleans and normalizes raw transcript segments before they are used
for summarization, chunking, and embeddings.

Design principle (per project spec): clean the text, but never
destroy information (timestamps, speaker labels, meaningful content)
that downstream retrieval might need.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class CleanedSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class TranscriptProcessor:
    """
    Applies a sequence of cleaning steps to raw transcript segments:
      1. Whitespace normalization
      2. Repeated-word removal (e.g. "and and" -> "and")
      3. Punctuation normalization
      4. Removal of empty / filler-only segments
      5. Basic transcription-artifact cleanup (e.g. stray "[inaudible]" markers)
    """

    ARTIFACT_PATTERNS = [
        r"\[.*?inaudible.*?\]",
        r"\[.*?music.*?\]",
        r"\[.*?applause.*?\]",
        r"\[.*?noise.*?\]",
    ]

    def __init__(self):
        self._artifact_regex = re.compile(
            "|".join(self.ARTIFACT_PATTERNS), flags=re.IGNORECASE
        )
        self._repeat_word_regex = re.compile(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Runs all text-level cleaning steps on a single string."""
        if not text:
            return ""

        cleaned = text
        cleaned = self._artifact_regex.sub("", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = self._repeat_word_regex.sub(r"\1", cleaned)
        cleaned = re.sub(r"\s+([.,?!])", r"\1", cleaned)
        cleaned = re.sub(r"([.,?!])(?=[^\s])", r"\1 ", cleaned)

        return cleaned.strip()

    def process_segments(self, raw_segments: List[dict]) -> List[CleanedSegment]:
        """
        Cleans a list of raw segment dicts (as loaded from Whisper's JSON output)
        and returns a list of CleanedSegment objects.
        """
        cleaned_segments: List[CleanedSegment] = []
        dropped_count = 0

        for seg in raw_segments:
            text = self.clean_text(seg.get("text", ""))

            if not text:
                dropped_count += 1
                continue

            cleaned_segments.append(
                CleanedSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=text,
                    speaker=seg.get("speaker"),
                )
            )

        if dropped_count:
            logger.info(f"Dropped {dropped_count} empty/artifact-only segment(s) during cleaning.")

        logger.info(f"Cleaned {len(cleaned_segments)} segments.")
        return cleaned_segments

    @staticmethod
    def load_raw_segments(json_path: str) -> List[dict]:
        """Loads raw Whisper-style segments from a JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript JSON not found: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_cleaned_segments(segments: List[CleanedSegment], output_path: str) -> None:
        """Saves cleaned segments to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in segments]
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved cleaned transcript to {out}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcript_processor.py <raw_transcript_json_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    processor = TranscriptProcessor()

    raw = processor.load_raw_segments(input_path)
    cleaned = processor.process_segments(raw)

    print("\n--- Cleaned Transcript ---")
    for seg in cleaned:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")

    stem = Path(input_path).stem
    processor.save_cleaned_segments(cleaned, f"data/processed/{stem}_cleaned.json")
