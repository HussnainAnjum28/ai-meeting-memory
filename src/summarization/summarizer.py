"""
summarizer.py

Generates a structured meeting summary (overview, key discussion points,
decisions, and action items) from a cleaned transcript, using the LLM
interface so it can work with Ollama or any other configured provider.

Per project spec: if a responsible person or deadline is not explicitly
stated, this must be marked as unknown rather than invented.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ActionItem:
    task: str
    assigned_to: str = "Unknown"
    deadline: str = "Unknown"
    evidence: str = ""


@dataclass
class MeetingSummary:
    overview: str
    key_points: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)


SUMMARY_PROMPT_TEMPLATE = """You are analyzing a meeting transcript. Based ONLY on the transcript below, produce a structured summary.

Respond with ONLY valid JSON (no markdown, no extra text) in exactly this format:
{{
  "overview": "2-3 sentence summary of what the meeting was about",
  "key_points": ["point 1", "point 2"],
  "decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"task": "...", "assigned_to": "Unknown or name", "deadline": "Unknown or date", "evidence": "short quote"}}
  ]
}}

Rules:
- If assigned_to or deadline is not explicitly stated in the transcript, use exactly "Unknown". Do not guess.
- Do not invent facts not present in the transcript.
- Keep it concise.

Transcript:
{transcript}

JSON:"""


class Summarizer:
    """
    Uses an LLM to produce a structured summary from a cleaned transcript.
    Includes defensive JSON parsing since LLMs sometimes wrap JSON in
    markdown fences or add stray text despite instructions.
    """

    def __init__(self, llm):
        """
        Args:
            llm: an instance of LLMInterface (rag/llm_interface.py)
        """
        self.llm = llm

    def summarize(self, cleaned_segments: List[dict]) -> MeetingSummary:
        """
        Args:
            cleaned_segments: list of cleaned transcript segment dicts.

        Returns:
            A MeetingSummary object. Falls back to a minimal summary
            (never raises) if the LLM output cannot be parsed, so the
            UI never crashes on a malformed model response.
        """
        if not cleaned_segments:
            return MeetingSummary(overview="No transcript content available to summarize.")

        transcript_text = "\n".join(seg["text"] for seg in cleaned_segments)

        prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript_text)

        logger.info("Generating meeting summary...")
        try:
            raw_response = self.llm.generate(prompt, temperature=0.1)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return MeetingSummary(overview=f"Summary could not be generated: {e}")

        return self._parse_summary(raw_response)

    def _parse_summary(self, raw_response: str) -> MeetingSummary:
        """Extracts and parses JSON from the LLM's raw text response."""
        json_str = self._extract_json(raw_response)

        if not json_str:
            logger.warning("Could not locate JSON in LLM response. Falling back to raw text overview.")
            return MeetingSummary(overview=raw_response.strip() or "Summary generation returned no content.")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse summary JSON: {e}. Falling back to raw text overview.")
            return MeetingSummary(overview=raw_response.strip())

        action_items = [
            ActionItem(
                task=item.get("task", "Unknown"),
                assigned_to=item.get("assigned_to", "Unknown"),
                deadline=item.get("deadline", "Unknown"),
                evidence=item.get("evidence", ""),
            )
            for item in data.get("action_items", [])
        ]

        return MeetingSummary(
            overview=data.get("overview", ""),
            key_points=data.get("key_points", []),
            decisions=data.get("decisions", []),
            action_items=action_items,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Pulls a JSON object out of text that may contain markdown fences or extra prose."""
        # Try to find a ```json ... ``` fenced block first
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        # Otherwise, find the first { ... } block (greedy to the last closing brace)
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        return None


if __name__ == "__main__":
    # Manual test runner.
    # Usage: python -m src.summarization.summarizer data/processed/sample_cleaned.json
    import sys
    sys.path.insert(0, ".")

    from src.rag.llm_interface import get_llm

    if len(sys.argv) < 2:
        print("Usage: python -m src.summarization.summarizer <cleaned_transcript_json_path>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        segments = json.load(f)

    llm = get_llm(provider="ollama", model_name="llama3.2")
    summarizer = Summarizer(llm)
    summary = summarizer.summarize(segments)

    print("\n=== OVERVIEW ===")
    print(summary.overview)

    print("\n=== KEY POINTS ===")
    for p in summary.key_points:
        print(f"- {p}")

    print("\n=== DECISIONS ===")
    for d in summary.decisions:
        print(f"- {d}")

    print("\n=== ACTION ITEMS ===")
    for a in summary.action_items:
        print(f"- Task: {a.task}")
        print(f"  Assigned to: {a.assigned_to}")
        print(f"  Deadline: {a.deadline}")
        print(f"  Evidence: {a.evidence}")
