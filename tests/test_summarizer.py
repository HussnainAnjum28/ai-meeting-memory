"""
test_summarizer.py

Unit tests for the summary JSON parsing logic. Uses a fake/mock LLM
so no real model call is needed — keeps tests instant and deterministic.
"""

import sys
sys.path.insert(0, ".")

from src.summarization.summarizer import Summarizer


class FakeLLM:
    """A stand-in LLM that returns whatever canned response we set."""
    def __init__(self, canned_response):
        self.canned_response = canned_response

    def generate(self, prompt, temperature=0.1):
        return self.canned_response


def test_parses_clean_json():
    fake_response = """{
        "overview": "A test meeting.",
        "key_points": ["point one", "point two"],
        "decisions": ["decision one"],
        "action_items": [
            {"task": "Do the thing", "assigned_to": "Unknown", "deadline": "Unknown", "evidence": "quote"}
        ]
    }"""
    summarizer = Summarizer(FakeLLM(fake_response))
    result = summarizer.summarize([{"text": "some transcript text"}])

    assert result.overview == "A test meeting."
    assert len(result.key_points) == 2
    assert len(result.action_items) == 1
    assert result.action_items[0].task == "Do the thing"


def test_parses_json_wrapped_in_markdown_fences():
    fake_response = """Here is the summary:
```json
{
    "overview": "Fenced meeting.",
    "key_points": [],
    "decisions": [],
    "action_items": []
}
```
Hope that helps!"""
    summarizer = Summarizer(FakeLLM(fake_response))
    result = summarizer.summarize([{"text": "some transcript text"}])

    assert result.overview == "Fenced meeting."


def test_falls_back_gracefully_on_invalid_json():
    fake_response = "This is not JSON at all, just plain text."
    summarizer = Summarizer(FakeLLM(fake_response))
    result = summarizer.summarize([{"text": "some transcript text"}])

    assert result.overview == "This is not JSON at all, just plain text."
    assert result.key_points == []
    assert result.action_items == []


def test_empty_transcript_returns_minimal_summary():
    summarizer = Summarizer(FakeLLM("irrelevant"))
    result = summarizer.summarize([])

    assert "No transcript content" in result.overview


def test_missing_fields_default_to_unknown():
    fake_response = """{
        "overview": "Minimal meeting.",
        "key_points": [],
        "decisions": [],
        "action_items": [
            {"task": "Something to do"}
        ]
    }"""
    summarizer = Summarizer(FakeLLM(fake_response))
    result = summarizer.summarize([{"text": "some transcript text"}])

    assert result.action_items[0].assigned_to == "Unknown"
    assert result.action_items[0].deadline == "Unknown"
