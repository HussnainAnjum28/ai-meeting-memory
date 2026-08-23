"""
test_chunker.py

Unit tests for transcript chunking logic (character-based chunking
with overlap). Uses small hand-crafted segment lists.
"""

import sys
sys.path.insert(0, ".")

from src.preprocessing.chunker import TranscriptChunker


def test_chunk_segments_respects_chunk_size():
    chunker = TranscriptChunker(chunk_size=50, chunk_overlap=10)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "This is the first sentence here.", "speaker": None},
        {"start": 1.0, "end": 2.0, "text": "This is the second sentence here.", "speaker": None},
        {"start": 2.0, "end": 3.0, "text": "This is the third sentence here.", "speaker": None},
    ]
    chunks = chunker.chunk_segments(segments, meeting_id="test")
    assert len(chunks) >= 2


def test_chunk_segments_preserves_meeting_id():
    chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": None}]
    chunks = chunker.chunk_segments(segments, meeting_id="my_meeting")
    assert chunks[0].meeting_id == "my_meeting"


def test_chunk_segments_empty_input_returns_empty_list():
    chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_segments([], meeting_id="test")
    assert chunks == []


def test_chunk_segments_single_speaker_preserved():
    chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "there", "speaker": "SPEAKER_00"},
    ]
    chunks = chunker.chunk_segments(segments, meeting_id="test")
    assert chunks[0].speaker == "SPEAKER_00"


def test_chunk_segments_mixed_speakers_results_in_none():
    chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "there", "speaker": "SPEAKER_01"},
    ]
    chunks = chunker.chunk_segments(segments, meeting_id="test")
    assert chunks[0].speaker is None


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    try:
        TranscriptChunker(chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError to be raised"
    except ValueError:
        pass
