"""
api_server.py

FastAPI backend exposing the AI Meeting Memory pipeline as a REST API,
so a custom frontend (or any client) can upload meetings, fetch
summaries/transcripts, and ask questions.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.audio.transcription import Transcriber
from src.preprocessing.transcript_processor import TranscriptProcessor
from src.preprocessing.chunker import TranscriptChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.rag.llm_interface import get_llm
from src.rag.qa_pipeline import RAGPipeline
from src.summarization.summarizer import Summarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Meeting Memory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: meeting_id -> meeting data (summary, transcript, filename)
MEETINGS: dict = {}

logger.info("Loading models (this happens once at server startup)...")
TRANSCRIBER = Transcriber(model_size="base", device="cpu", compute_type="int8")
EMBEDDER = Embedder(model_name="all-MiniLM-L6-v2")
VECTOR_STORE = VectorStore()
LLM = get_llm(provider="ollama", model_name="llama3.2")
RAG_PIPELINE = RAGPipeline(vector_store=VECTOR_STORE, llm=LLM)
SUMMARIZER = Summarizer(LLM)
logger.info("All models loaded. API ready.")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/meetings")
def list_meetings():
    """Returns a summary list of all processed meetings."""
    return [
        {
            "meeting_id": mid,
            "filename": m["filename"],
            "status": m["status"],
        }
        for mid, m in MEETINGS.items()
    ]


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    """Returns full details (summary + transcript) for one meeting."""
    if meeting_id not in MEETINGS:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting = MEETINGS[meeting_id]
    if meeting["status"] == "error":
        return {"meeting_id": meeting_id, "status": "error", "error": meeting.get("error")}

    summary = meeting["summary"]
    return {
        "meeting_id": meeting_id,
        "status": "ready",
        "filename": meeting["filename"],
        "transcript": meeting["cleaned_segments"],
        "summary": {
            "overview": summary.overview,
            "key_points": summary.key_points,
            "decisions": summary.decisions,
            "action_items": [
                {
                    "task": a.task,
                    "assigned_to": a.assigned_to,
                    "deadline": a.deadline,
                    "evidence": a.evidence,
                }
                for a in summary.action_items
            ],
        },
    }


@app.post("/meetings/upload")
async def upload_meeting(file: UploadFile = File(...)):
    """
    Runs the full pipeline on an uploaded audio file: transcribe -> clean
    -> chunk -> embed -> store -> summarize. Returns the meeting_id.
    """
    meeting_id = Path(file.filename).stem

    raw_path = Path("data/raw") / file.filename
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    contents = await file.read()
    with open(raw_path, "wb") as f:
        f.write(contents)

    try:
        logger.info(f"Transcribing {file.filename}...")
        segments = TRANSCRIBER.transcribe(str(raw_path))
        raw_segments_dicts = [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in segments
        ]

        if not raw_segments_dicts:
            MEETINGS[meeting_id] = {
                "status": "error",
                "error": "Empty or silent audio file.",
                "filename": file.filename,
            }
            return {"meeting_id": meeting_id, "status": "error", "error": "Empty or silent audio file."}

        logger.info("Cleaning transcript...")
        processor = TranscriptProcessor()
        cleaned = processor.process_segments(raw_segments_dicts)
        cleaned_dicts = [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in cleaned
        ]

        logger.info("Chunking transcript...")
        chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_segments(cleaned_dicts, meeting_id=meeting_id)
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id, "text": c.text, "start": c.start,
                "end": c.end, "speaker": c.speaker, "meeting_id": c.meeting_id,
            }
            for c in chunks
        ]

        logger.info("Generating embeddings...")
        embedded = EMBEDDER.embed_chunks(chunk_dicts)
        embedded_dicts = [
            {
                "chunk_id": e.chunk_id, "text": e.text, "start": e.start, "end": e.end,
                "speaker": e.speaker, "meeting_id": e.meeting_id, "embedding": e.embedding,
            }
            for e in embedded
        ]

        logger.info("Storing in vector database...")
        VECTOR_STORE.add_embedded_chunks(embedded_dicts)

        logger.info("Generating summary...")
        summary = SUMMARIZER.summarize(cleaned_dicts)

        MEETINGS[meeting_id] = {
            "status": "ready",
            "cleaned_segments": cleaned_dicts,
            "summary": summary,
            "filename": file.filename,
        }

        logger.info(f"Meeting '{meeting_id}' processed successfully.")
        return {"meeting_id": meeting_id, "status": "ready"}

    except Exception as e:
        logger.exception("Pipeline failed")
        MEETINGS[meeting_id] = {"status": "error", "error": str(e), "filename": file.filename}
        return {"meeting_id": meeting_id, "status": "error", "error": str(e)}


@app.post("/meetings/{meeting_id}/ask")
def ask_question(meeting_id: str, request: AskRequest):
    """Answers a question about a specific meeting using RAG."""
    if meeting_id not in MEETINGS or MEETINGS[meeting_id]["status"] != "ready":
        raise HTTPException(status_code=404, detail="Meeting not found or not ready")

    result = RAG_PIPELINE.answer(request.question, meeting_id=meeting_id)
    return {
        "answer": result.answer,
        "evidence": [
            {"text": e.text, "start": e.start, "end": e.end, "speaker": e.speaker}
            for e in result.evidence
        ],
    }
