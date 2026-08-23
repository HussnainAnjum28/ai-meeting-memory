"""
test_transcript_processor.py

Unit tests for the transcript cleaning logic. Uses small, hand-crafted
fixtures (no real audio or models needed) so these run instantly.
"""

import sys
sys.path.insert(0, ".")

from src.preprocessing.transcript_processor import TranscriptProcessor


def test_clean_text_removes_repeated_words():
    processor = TranscriptProcessor()
    result = processor.clean_text("Thanks, and and I can help")
    assert "and and" not in result
    assert result == "Thanks, and I can help"


def test_clean_text_removes_artifacts():
    processor = TranscriptProcessor()
    result = processor.clean_text("Hello [inaudible] there")
    assert "[inaudible]" not in result


def test_clean_text_collapses_whitespace():
    processor = TranscriptProcessor()
    result = processor.clean_text("Hello    there   friend")
    assert result == "Hello there friend"


def test_clean_text_handles_empty_string():
    processor = TranscriptProcessor()
    result = processor.clean_text("")
    assert result == ""


def test_process_segments_drops_empty_segments():
    processor = TranscriptProcessor()
    raw_segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello there", "speaker": None},
        {"start": 1.0, "end": 2.0, "text": "[inaudible]", "speaker": None},
        {"start": 2.0, "end": 3.0, "text": "Goodbye", "speaker": None},
    ]
    cleaned = processor.process_segments(raw_segments)
    assert len(cleaned) == 2
    assert cleaned[0].text == "Hello there"
    assert cleaned[1].text == "Goodbye"


def test_process_segments_preserves_timestamps():
    processor = TranscriptProcessor()
    raw_segments = [{"start": 5.5, "end": 10.2, "text": "Testing", "speaker": "SPEAKER_00"}]
    cleaned = processor.process_segments(raw_segments)
    assert cleaned[0].start == 5.5
    assert cleaned[0].end == 10.2
    assert cleaned[0].speaker == "SPEAKER_00"
