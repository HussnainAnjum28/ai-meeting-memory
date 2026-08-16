"""
diarization.py

Speaker diarization using pyannote.audio: determines "who spoke when"
in an audio file, independent of transcription.

Audio is pre-converted to 16kHz mono WAV using the system ffmpeg CLI
and loaded manually with soundfile, bypassing pyannote's default
torchcodec-based audio loader (which has known DLL-loading issues
on Windows).
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class Diarizer:
    def __init__(self, hf_token: str = None):
        from pyannote.audio import Pipeline

        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not set. Add it to your .env file.")

        logger.info("Loading speaker diarization model (first run downloads it, this can take a while)...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        logger.info("Diarization model loaded.")

    def _load_waveform(self, audio_path: str):
        """Converts audio to 16kHz mono WAV via ffmpeg CLI, then loads it."""
        import soundfile as sf
        import torch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", wav_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")

            data, sample_rate = sf.read(wav_path, dtype="float32")
            waveform = torch.from_numpy(data).unsqueeze(0)
            return waveform, sample_rate
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def diarize(self, audio_path: str) -> List[SpeakerTurn]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Running diarization on {audio_path}...")
        waveform, sample_rate = self._load_waveform(str(path))

        result = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # Newer pyannote versions wrap the Annotation in a DiarizeOutput object;
        # older versions return the Annotation directly. Handle both.
        if hasattr(result, "itertracks"):
            annotation = result
        elif hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        elif hasattr(result, "annotation"):
            annotation = result.annotation
        else:
            raise AttributeError(f"Unexpected diarization result type: {type(result)}")

        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(SpeakerTurn(start=round(turn.start, 2), end=round(turn.end, 2), speaker=speaker))

        logger.info(f"Diarization found {len(set(t.speaker for t in turns))} speaker(s), {len(turns)} turn(s).")
        return turns

    @staticmethod
    def assign_speakers_to_segments(segments: List[dict], speaker_turns: List[SpeakerTurn]) -> List[dict]:
        if not speaker_turns:
            return segments

        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]
            best_overlap = 0.0
            best_speaker = None

            for turn in speaker_turns:
                overlap = min(seg_end, turn.end) - max(seg_start, turn.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker

            if best_speaker:
                seg["speaker"] = best_speaker

        return segments


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python -m src.audio.diarization <audio_file_path>")
        sys.exit(1)

    diarizer = Diarizer()
    turns = diarizer.diarize(sys.argv[1])

    print("\n--- Speaker Turns ---")
    for t in turns:
        print(f"[{t.start:.1f}s - {t.end:.1f}s] {t.speaker}")

