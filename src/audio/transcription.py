"""
transcription.py

Handles converting audio files into timestamped transcripts using
faster-whisper (an efficient re-implementation of OpenAI's Whisper).

This module is intentionally kept independent from the rest of the
pipeline so the speech-to-text engine can be swapped later without
touching downstream code.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json
import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TranscriptSegment:
    """A single chunk of transcribed speech with timing metadata."""
    start: float          # seconds
    end: float            # seconds
    text: str
    speaker: Optional[str] = None  # left None until diarization is added


class Transcriber:
    """
    Wraps faster-whisper to turn an audio file into a list of
    TranscriptSegment objects.

    Model sizes (accuracy vs speed): tiny, base, small, medium, large-v3
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Args:
            model_size: Whisper model size to load.
            device: "cpu" or "cuda" (if a GPU + CUDA is available).
            compute_type: Precision used for inference. "int8" is fast
                and memory-efficient on CPU.
        """
        logger.info(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Model loaded successfully.")

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        """
        Transcribes an audio file into timestamped segments.

        Args:
            audio_path: Path to an audio file (mp3, wav, m4a, etc.)

        Returns:
            A list of TranscriptSegment objects.

        Raises:
            FileNotFoundError: if the audio file does not exist.
            RuntimeError: if transcription fails for any other reason.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing: {audio_path}")

        try:
            segments_iter, info = self.model.transcribe(str(path), beam_size=5)
        except Exception as e:
            raise RuntimeError(f"Transcription failed for {audio_path}: {e}") from e

        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        segments: List[TranscriptSegment] = []
        for seg in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg.text.strip(),
                    speaker=None,  # Speaker diarization not yet implemented
                )
            )

        if not segments:
            logger.warning("Transcription produced no segments (empty or silent audio?).")

        return segments

    @staticmethod
    def to_readable_text(segments: List[TranscriptSegment]) -> str:
        """Formats segments into the human-readable [HH:MM:SS] style transcript."""
        lines = []
        for seg in segments:
            timestamp = Transcriber._format_timestamp(seg.start)
            speaker_label = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"[{timestamp}] {speaker_label}{seg.text}")
        return "\n".join(lines)

    @staticmethod
    def to_json(segments: List[TranscriptSegment]) -> str:
        """Serializes segments to a JSON string for storage."""
        return json.dumps([asdict(s) for s in segments], indent=2, ensure_ascii=False)

    @staticmethod
    def save_transcript(segments: List[TranscriptSegment], output_path: str) -> None:
        """Saves both a JSON and a readable .txt version of the transcript."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        json_path = out.with_suffix(".json")
        txt_path = out.with_suffix(".txt")

        json_path.write_text(Transcriber.to_json(segments), encoding="utf-8")
        txt_path.write_text(Transcriber.to_readable_text(segments), encoding="utf-8")

        logger.info(f"Saved transcript to {json_path} and {txt_path}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Converts seconds -> HH:MM:SS string."""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    # Simple manual test runner.
    # Usage: python transcription.py path/to/audio.mp3
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcription.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcriber = Transcriber(model_size="base", device="cpu", compute_type="int8")
    result_segments = transcriber.transcribe(audio_file)

    print("\n--- Readable Transcript ---")
    print(Transcriber.to_readable_text(result_segments))

    output_name = Path(audio_file).stem
    Transcriber.save_transcript(result_segments, f"data/processed/{output_name}")
