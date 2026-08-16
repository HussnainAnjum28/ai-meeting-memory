"""
streamlit_app.py

Main Streamlit UI for AI Meeting Memory.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

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

st.set_page_config(page_title="AI Meeting Memory", page_icon=":brain:", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    letter-spacing: -0.01em;
}

/* Base surface */
.stApp {
    background-color: #0d0e14;
}

/* Sidebar: subtle, not pitch black, quiet border */
section[data-testid="stSidebar"] {
    background-color: #111219;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 24px;
}

/* Sidebar brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 24px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sidebar-brand .mark {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #1c1e2b;
    border: 1px solid rgba(255,255,255,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}
.sidebar-brand .title {
    font-size: 15px;
    font-weight: 600;
    color: #e8e8ec;
}
.sidebar-brand .subtitle {
    font-size: 11px;
    color: #6b6d7a;
    margin-top: -2px;
}

/* Section labels in sidebar */
.side-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b6d7a;
    margin: 24px 0 8px 0;
}

/* Header: restrained, not a giant gradient banner */
.app-header {
    padding: 24px 0 24px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    animation: fadeInUp 300ms ease-out;
}
.app-header h1 {
    color: #f2f2f5;
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}
.app-header p {
    color: #8b8d9a;
    margin: 6px 0 0 0;
    font-size: 13.5px;
}

/* Cards: subtle border + soft shadow, gentle hover lift */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    background: #14151d !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.4);
    transition: border-color 200ms ease-out, transform 200ms ease-out, box-shadow 200ms ease-out;
    animation: fadeInUp 250ms ease-out;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255,255,255,0.14) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 500;
    font-size: 14px;
    color: #8b8d9a;
    transition: color 150ms ease-out;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e8e8ec !important;
    font-weight: 600;
}
div[data-baseweb="tab-highlight"] {
    background-color: #7c6df2 !important;
    height: 2px !important;
}

/* Buttons */
.stButton button, .stFormSubmitButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    transition: transform 150ms ease-out, filter 150ms ease-out !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    filter: brightness(1.08);
}
.stButton button:active, .stFormSubmitButton button:active {
    transform: scale(0.98);
}
.stButton button[kind="primary"] {
    background: #6f5ce0 !important;
    border: none !important;
}
.stButton button[kind="primary"]:hover {
    background: #7d6bea !important;
}

/* Badges: quiet pill, subtle border instead of loud fill */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    color: #b4b6c2;
}
.badge-accent { color: #ab9dfa; border-color: rgba(171,157,250,0.25); background: rgba(171,157,250,0.06); }

/* Timestamp chips */
.ts-chip {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    color: #9a9cae;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 1px 7px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 8px;
    font-family: 'SF Mono', 'Consolas', monospace;
}

/* File uploader + inputs: quiet focus ring */
.stFileUploader, .stChatInput textarea, .stSelectbox div[data-baseweb="select"] {
    border-radius: 8px !important;
}

/* Chat input focus */
.stChatInput textarea:focus {
    box-shadow: 0 0 0 2px rgba(124,109,242,0.35) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "meetings" not in st.session_state:
    st.session_state.meetings = {}
if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}


@st.cache_resource
def load_heavy_resources():
    transcriber = Transcriber(model_size="base", device="cpu", compute_type="int8")
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore()
    llm = get_llm(provider="ollama", model_name="llama3.2")
    rag_pipeline = RAGPipeline(vector_store=vector_store, llm=llm)
    summarizer = Summarizer(llm)
    return {
        "transcriber": transcriber,
        "embedder": embedder,
        "vector_store": vector_store,
        "llm": llm,
        "rag_pipeline": rag_pipeline,
        "summarizer": summarizer,
    }


def process_meeting(uploaded_file, resources) -> str:
    meeting_id = Path(uploaded_file.name).stem

    raw_path = Path("data/raw") / uploaded_file.name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    status = st.status(f"Processing '{uploaded_file.name}'...", expanded=True)

    try:
        status.write("Transcribing audio (this may take a while)...")
        segments = resources["transcriber"].transcribe(str(raw_path))
        raw_segments_dicts = [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in segments
        ]

        if not raw_segments_dicts:
            status.update(label="Transcription produced no content.", state="error")
            st.session_state.meetings[meeting_id] = {
                "status": "error",
                "error": "Empty or silent audio file.",
            }
            return meeting_id

        status.write("Cleaning transcript...")
        processor = TranscriptProcessor()
        cleaned = processor.process_segments(raw_segments_dicts)
        cleaned_dicts = [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker}
            for s in cleaned
        ]

        status.write("Chunking transcript...")
        chunker = TranscriptChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_segments(cleaned_dicts, meeting_id=meeting_id)
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id, "text": c.text, "start": c.start,
                "end": c.end, "speaker": c.speaker, "meeting_id": c.meeting_id,
            }
            for c in chunks
        ]

        status.write("Generating embeddings...")
        embedded = resources["embedder"].embed_chunks(chunk_dicts)
        embedded_dicts = [
            {
                "chunk_id": e.chunk_id, "text": e.text, "start": e.start, "end": e.end,
                "speaker": e.speaker, "meeting_id": e.meeting_id, "embedding": e.embedding,
            }
            for e in embedded
        ]

        status.write("Storing in vector database...")
        resources["vector_store"].add_embedded_chunks(embedded_dicts)

        status.write("Generating summary...")
        summary = resources["summarizer"].summarize(cleaned_dicts)

        st.session_state.meetings[meeting_id] = {
            "status": "ready",
            "cleaned_segments": cleaned_dicts,
            "summary": summary,
            "filename": uploaded_file.name,
        }
        st.session_state.chat_history[meeting_id] = []

        status.update(label=f"'{uploaded_file.name}' processed successfully!", state="complete")

    except Exception as e:
        logger.exception("Pipeline failed")
        status.update(label=f"Processing failed: {e}", state="error")
        st.session_state.meetings[meeting_id] = {"status": "error", "error": str(e)}

    return meeting_id


def render_sidebar(resources):
    with st.sidebar:
        brand_html = '<div class="sidebar-brand"><div class="mark">AI</div><div><div class="title">Meeting Memory</div><div class="subtitle">Local AI transcript assistant</div></div></div>'
        st.markdown(brand_html, unsafe_allow_html=True)

        st.markdown('<div class="side-label">Upload Meeting</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an audio file", type=["mp3", "wav", "m4a"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            if st.button("Process Meeting", type="primary", use_container_width=True):
                meeting_id = process_meeting(uploaded_file, resources)
                st.session_state.current_meeting_id = meeting_id
                st.rerun()

        st.markdown('<div class="side-label">Select Meeting</div>', unsafe_allow_html=True)
        ready_meetings = [
            mid for mid, m in st.session_state.meetings.items() if m.get("status") == "ready"
        ]
        if ready_meetings:
            selected = st.selectbox(
                "Meetings",
                options=ready_meetings,
                format_func=lambda mid: st.session_state.meetings[mid].get("filename", mid),
                index=ready_meetings.index(st.session_state.current_meeting_id)
                if st.session_state.current_meeting_id in ready_meetings else 0,
                label_visibility="collapsed",
            )
            st.session_state.current_meeting_id = selected
        else:
            st.caption("No processed meetings yet.")

        st.markdown('<div class="side-label">Model Settings</div>', unsafe_allow_html=True)
        st.markdown('<span class="badge">Whisper base</span><span class="badge">MiniLM embeddings</span><span class="badge badge-accent">llama3.2 local</span>', unsafe_allow_html=True)
        st.caption("All models run locally. No data leaves your machine.")


def render_main_page(resources):
    header_html = '<div class="app-header"><h1>AI Meeting Memory</h1><p>Upload a meeting, get instant summaries, decisions, and action items, and ask it anything.</p></div>'
    st.markdown(header_html, unsafe_allow_html=True)

    meeting_id = st.session_state.current_meeting_id

    if not meeting_id or meeting_id not in st.session_state.meetings:
        st.info("Upload a meeting audio file from the sidebar to get started.")
        return

    meeting = st.session_state.meetings[meeting_id]

    if meeting.get("status") == "error":
        st.error(f"This meeting failed to process: {meeting.get('error', 'Unknown error')}")
        return

    st.subheader(meeting.get("filename", meeting_id))

    tab_summary, tab_transcript, tab_chat = st.tabs(["Summary", "Transcript", "Ask Your Meeting"])

    with tab_summary:
        summary = meeting["summary"]

        with st.container(border=True):
            st.markdown("##### Overview")
            st.write(summary.overview or "No overview available.")

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("##### Key Discussion Points")
                if summary.key_points:
                    for p in summary.key_points:
                        st.markdown(f"- {p}")
                else:
                    st.caption("No key points extracted.")

        with col2:
            with st.container(border=True):
                st.markdown("##### Decisions")
                if summary.decisions:
                    for d in summary.decisions:
                        st.markdown(f"- {d}")
                else:
                    st.caption("No decisions extracted.")

        st.markdown("##### Action Items")
        if summary.action_items:
            for item in summary.action_items:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{item.task}**")
                        if item.evidence:
                            st.caption(f"Evidence: {item.evidence}")
                    with c2:
                        st.markdown(f'<span class="badge">Assigned: {item.assigned_to}</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="badge">Due: {item.deadline}</span>', unsafe_allow_html=True)
        else:
            st.caption("No action items extracted.")

    with tab_transcript:
        with st.container(border=True):
            for seg in meeting["cleaned_segments"]:
                mins, secs = divmod(int(seg["start"]), 60)
                speaker = f"**{seg['speaker']}:** " if seg.get("speaker") else ""
                line_html = f'<span class="ts-chip">{mins:02d}:{secs:02d}</span> {speaker}{seg["text"]}'
                st.markdown(line_html, unsafe_allow_html=True)
                st.markdown("")

    with tab_chat:
        history = st.session_state.chat_history.setdefault(meeting_id, [])

        for question, answer, evidence in history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
                if evidence:
                    with st.expander("Evidence"):
                        for e in evidence:
                            mins, secs = divmod(int(e.start), 60)
                            ev_html = f'<span class="ts-chip">{mins:02d}:{secs:02d}</span> {e.text}'
                            st.markdown(ev_html, unsafe_allow_html=True)

        question = st.chat_input("Ask something about this meeting...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = resources["rag_pipeline"].answer(question, meeting_id=meeting_id)
                        st.write(result.answer)
                        if result.evidence:
                            with st.expander("Evidence"):
                                for e in result.evidence:
                                    mins, secs = divmod(int(e.start), 60)
                                    ev_html = f'<span class="ts-chip">{mins:02d}:{secs:02d}</span> {e.text}'
                                    st.markdown(ev_html, unsafe_allow_html=True)
                        history.append((question, result.answer, result.evidence))
                    except Exception as e:
                        error_msg = f"Sorry, I could not answer that: {e}"
                        st.error(error_msg)
                        history.append((question, error_msg, []))


def main():
    resources = load_heavy_resources()
    render_sidebar(resources)
    render_main_page(resources)


if __name__ == "__main__":
    main()
