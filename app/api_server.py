"""
api_server.py

FastAPI backend exposing the AI Meeting Memory pipeline as a REST API.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.audio.transcription import Transcriber
from src.preprocessing.transcript_processor import TranscriptProcessor
from src.preprocessing.chunker import TranscriptChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.rag.llm_interface import get_llm
from src.rag.qa_pipeline import RAGPipeline
from src.summarization.summarizer import Summarizer
from src.audio.diarization import Diarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Meeting Memory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MEETINGS: dict = {}

logger.info("Loading models (this happens once at server startup)...")
TRANSCRIBER = Transcriber(model_size="base", device="cpu", compute_type="int8")
EMBEDDER = Embedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")
VECTOR_STORE = VectorStore()
LLM = get_llm(provider="ollama", model_name="llama3.2")
RAG_PIPELINE = RAGPipeline(vector_store=VECTOR_STORE, llm=LLM, embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2")
SUMMARIZER = Summarizer(LLM)

try:
    DIARIZER = Diarizer()
    DIARIZATION_ENABLED = True
    logger.info("Speaker diarization enabled.")
except Exception as e:
    DIARIZER = None
    DIARIZATION_ENABLED = False
    logger.warning(f"Speaker diarization disabled (could not load): {e}")
logger.info("All models loaded. API ready.")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/meetings")
def list_meetings():
    return [
        {"meeting_id": mid, "filename": m["filename"], "status": m["status"]}
        for mid, m in MEETINGS.items()
    ]


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
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
                {"task": a.task, "assigned_to": a.assigned_to, "deadline": a.deadline, "evidence": a.evidence}
                for a in summary.action_items
            ],
        },
    }


@app.post("/meetings/upload")
async def upload_meeting(file: UploadFile = File(...), enable_diarization: bool = Form(True)):
    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Please upload an MP3, WAV, M4A, or WEBM audio file.",
        )

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
            MEETINGS[meeting_id] = {"status": "error", "error": "Empty or silent audio file.", "filename": file.filename}
            return {"meeting_id": meeting_id, "status": "error", "error": "Empty or silent audio file."}

        if DIARIZATION_ENABLED and enable_diarization:
            try:
                logger.info("Running speaker diarization...")
                speaker_turns = DIARIZER.diarize(str(raw_path))
                raw_segments_dicts = Diarizer.assign_speakers_to_segments(raw_segments_dicts, speaker_turns)
            except Exception as e:
                logger.warning(f"Diarization failed, continuing without speaker labels: {e}")

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
            {"chunk_id": c.chunk_id, "text": c.text, "start": c.start, "end": c.end, "speaker": c.speaker, "meeting_id": c.meeting_id}
            for c in chunks
        ]

        logger.info("Generating embeddings...")
        embedded = EMBEDDER.embed_chunks(chunk_dicts)
        embedded_dicts = [
            {"chunk_id": e.chunk_id, "text": e.text, "start": e.start, "end": e.end, "speaker": e.speaker, "meeting_id": e.meeting_id, "embedding": e.embedding}
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
    if meeting_id not in MEETINGS or MEETINGS[meeting_id]["status"] != "ready":
        raise HTTPException(status_code=404, detail="Meeting not found or not ready")

    result = RAG_PIPELINE.answer(request.question, meeting_id=meeting_id)
    return {
        "answer": result.answer,
        "evidence": [{"text": e.text, "start": e.start, "end": e.end, "speaker": e.speaker} for e in result.evidence],
    }


def _mom_line(pdf, text, size=11, style="", gap=6):
    pdf.set_font("helvetica", style, size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, gap, text, align="L")


@app.get("/meetings/{meeting_id}/export/pdf")
def export_pdf(meeting_id: str):
    """Generates a professionally formatted Minutes of Meeting PDF."""
    from fpdf import FPDF

    if meeting_id not in MEETINGS or MEETINGS[meeting_id]["status"] != "ready":
        raise HTTPException(status_code=404, detail="Meeting not found or not ready")

    meeting = MEETINGS[meeting_id]
    summary = meeting["summary"]

    pdf = FPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()

    # Title block
    pdf.set_font("helvetica", "B", 18)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 10, "MINUTES OF MEETING", align="C")
    pdf.ln(2)

    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.cell(35, 7, "Meeting Title:")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 7, meeting["filename"], new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.cell(35, 7, "Date Generated:")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 7, datetime.now().strftime("%d %B %Y, %H:%M"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "B", 11)
    pdf.set_x(pdf.l_margin)
    pdf.cell(35, 7, "Prepared By:")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 7, "AI Meeting Memory (automated)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Section 1: Overview
    pdf.set_font("helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "1. Meeting Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(220, 220, 220)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    _mom_line(pdf, summary.overview or "No overview available.", size=11)
    pdf.ln(5)

    # Section 2: Key Discussion Points
    pdf.set_font("helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "2. Key Discussion Points", new_x="LMARGIN", new_y="NEXT")
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    if summary.key_points:
        for i, p in enumerate(summary.key_points, 1):
            _mom_line(pdf, f"{i}. {p}", size=11)
    else:
        _mom_line(pdf, "None recorded.", size=11)
    pdf.ln(5)

    # Section 3: Decisions
    pdf.set_font("helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "3. Decisions Made", new_x="LMARGIN", new_y="NEXT")
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    if summary.decisions:
        for i, d in enumerate(summary.decisions, 1):
            _mom_line(pdf, f"{i}. {d}", size=11)
    else:
        _mom_line(pdf, "None recorded.", size=11)
    pdf.ln(5)

    # Section 4: Action Items (table)
    pdf.set_font("helvetica", "B", 13)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 8, "4. Action Items", new_x="LMARGIN", new_y="NEXT")
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    if summary.action_items:
        col_task = pdf.epw * 0.5
        col_owner = pdf.epw * 0.25
        col_deadline = pdf.epw * 0.25

        pdf.set_font("helvetica", "B", 10)
        pdf.set_fill_color(235, 235, 240)
        pdf.set_x(pdf.l_margin)
        pdf.cell(col_task, 8, "Task", border=1, fill=True)
        pdf.cell(col_owner, 8, "Assigned To", border=1, fill=True)
        pdf.cell(col_deadline, 8, "Deadline", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 10)
        for item in summary.action_items:
            start_y = pdf.get_y()
            x_start = pdf.l_margin

            pdf.set_xy(x_start, start_y)
            pdf.multi_cell(col_task, 7, item.task, border=1)
            task_end_y = pdf.get_y()

            pdf.set_xy(x_start + col_task, start_y)
            pdf.multi_cell(col_owner, 7, item.assigned_to, border=1)
            owner_end_y = pdf.get_y()

            pdf.set_xy(x_start + col_task + col_owner, start_y)
            pdf.multi_cell(col_deadline, 7, item.deadline, border=1)
            deadline_end_y = pdf.get_y()

            row_end_y = max(task_end_y, owner_end_y, deadline_end_y)
            pdf.set_xy(x_start, row_end_y)
    else:
        _mom_line(pdf, "No action items recorded.", size=11)

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")

    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={meeting_id}_minutes.pdf"},
    )






@app.get("/meetings/{meeting_id}/analytics")
def get_analytics(meeting_id: str):
    """Computes analytics from the meeting transcript, including
    per-speaker talk-time breakdown when speaker labels are available."""
    import re
    from collections import Counter

    if meeting_id not in MEETINGS or MEETINGS[meeting_id]["status"] != "ready":
        raise HTTPException(status_code=404, detail="Meeting not found or not ready")

    segments = MEETINGS[meeting_id]["cleaned_segments"]

    if not segments:
        return {
            "meeting_id": meeting_id, "duration_seconds": 0, "word_count": 0,
            "words_per_minute": 0, "segment_count": 0, "top_keywords": [],
            "activity_buckets": [], "speaker_stats": [],
        }

    duration = segments[-1]["end"] - segments[0]["start"]
    full_text = " ".join(s["text"] for s in segments)
    words = re.findall(r"[a-zA-Z']+", full_text.lower())
    word_count = len(words)
    words_per_minute = round(word_count / (duration / 60), 1) if duration > 0 else 0

    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "on", "at", "for", "with", "as", "it", "this", "that", "we",
        "i", "you", "he", "she", "they", "them", "his", "her", "our", "your", "their",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "can", "so", "if", "not", "no", "yes", "just", "like", "about", "some", "there",
        "here", "what", "when", "where", "who", "how", "why", "which", "us", "im",
        "its", "also", "one", "get", "got", "going", "go", "really", "well", "know",
        "think", "good", "want", "need", "up", "out", "all", "been", "s", "t", "re", "ll",
        "ve", "d", "m",
    }
    meaningful_words = [w for w in words if w not in stopwords and len(w) > 2]
    top_keywords = [{"word": w, "count": c} for w, c in Counter(meaningful_words).most_common(10)]

    bucket_count = 5
    bucket_size = duration / bucket_count if duration > 0 else 1
    bucket_word_counts = [0] * bucket_count
    start_time = segments[0]["start"]
    for seg in segments:
        seg_words = len(re.findall(r"[a-zA-Z']+", seg["text"]))
        bucket_idx = min(int((seg["start"] - start_time) / bucket_size), bucket_count - 1)
        bucket_word_counts[bucket_idx] += seg_words

    activity_buckets = [
        {"label": f"Part {i+1}", "word_count": bucket_word_counts[i]}
        for i in range(bucket_count)
    ]

    speaker_time = {}
    speaker_words = {}
    for seg in segments:
        speaker = seg.get("speaker") or "Unknown"
        seg_duration = seg["end"] - seg["start"]
        seg_words = len(re.findall(r"[a-zA-Z']+", seg["text"]))
        speaker_time[speaker] = speaker_time.get(speaker, 0) + seg_duration
        speaker_words[speaker] = speaker_words.get(speaker, 0) + seg_words

    total_speaking_time = sum(speaker_time.values()) or 1
    speaker_stats = [
        {
            "speaker": spk,
            "seconds": round(t, 1),
            "percentage": round((t / total_speaking_time) * 100, 1),
            "word_count": speaker_words.get(spk, 0),
        }
        for spk, t in sorted(speaker_time.items(), key=lambda x: -x[1])
    ]

    return {
        "meeting_id": meeting_id,
        "duration_seconds": round(duration, 1),
        "word_count": word_count,
        "words_per_minute": words_per_minute,
        "segment_count": len(segments),
        "top_keywords": top_keywords,
        "activity_buckets": activity_buckets,
        "speaker_stats": speaker_stats,
    }


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/search")
def search_all_meetings(request: SearchRequest):
    """Searches across ALL processed meetings using semantic similarity,
    without restricting to a single meeting_id."""
    query_embedding = RAG_PIPELINE.embed_model.encode(request.query).tolist()
    results = VECTOR_STORE.query(query_embedding, top_k=request.top_k, meeting_id=None)

    enriched_results = []
    for r in results:
        meta = r["metadata"]
        mid = meta.get("meeting_id", "unknown")
        filename = MEETINGS.get(mid, {}).get("filename", mid)
        enriched_results.append({
            "meeting_id": mid,
            "filename": filename,
            "text": r["text"],
            "start": meta["start"],
            "end": meta["end"],
            "distance": r["distance"],
        })

    return {"query": request.query, "results": enriched_results}









# Serve the frontend (HTML/CSS/JS) as static files.
# Mounted last so it does not shadow the API routes above.
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="frontend")
