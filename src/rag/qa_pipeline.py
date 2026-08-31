"""
qa_pipeline.py

The core Retrieval-Augmented Generation (RAG) pipeline. Ties together
the embedder, vector store, and LLM to answer questions grounded in
actual meeting transcript evidence.

Anti-hallucination strategy (per project spec): the LLM is instructed
to answer ONLY from the provided context, and to explicitly say when
the answer is not present, rather than guessing.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Evidence:
    text: str
    start: float
    end: float
    speaker: Optional[str]
    distance: float


@dataclass
class QAResult:
    answer: str
    evidence: List[Evidence]
    question: str


RAG_PROMPT_TEMPLATE = """You are a professional meeting assistant. Your job is to help the user understand what was discussed in this meeting by answering their questions clearly and confidently, using only the transcript excerpts provided below.\n\nLanguage rule: Respond in the same language the user's question is written in (English, Urdu, or Roman Urdu). If the question is in Urdu or Roman Urdu, write your entire answer fluently in Urdu script, even if the transcript excerpts themselves are in English.\n\nConciseness rule: Keep your answer to 2-4 sentences maximum. Do not repeat the same phrase or idea more than once. If you notice yourself repeating a word or phrase, stop and move to the next point instead.

Guidelines:
- Write in a natural, helpful, professional tone, as a knowledgeable colleague would.
- Base your answer only on the information in the excerpts below. Do not invent names, deadlines, decisions, or any fact not stated there.
- If the excerpts do not contain the answer, say so plainly: "I couldn't find that in this meeting's transcript." Do not guess.
- Be concise but complete. Use short paragraphs or a bulleted list if it helps clarity.
- Do not mention "the excerpts" or "the transcript excerpts" in your answer, just answer as if you attended the meeting.

Meeting transcript excerpts:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Orchestrates the full RAG flow: embed question -> retrieve chunks
    -> build grounded prompt -> generate answer -> return with evidence.
    """

    def __init__(self, vector_store, llm, embedding_model_name: str = "all-MiniLM-L6-v2", top_k: int = 5):
        """
        Args:
            vector_store: an instance of VectorStore (retrieval/vector_store.py)
            llm: an instance of LLMInterface (rag/llm_interface.py)
            embedding_model_name: must match the model used to embed the chunks
            top_k: number of chunks to retrieve per question
        """
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        logger.info(f"Loading query embedding model '{embedding_model_name}'...")
        self.embed_model = SentenceTransformer(embedding_model_name)

    GREETINGS = {
        "hi", "hello", "hey", "hiya", "yo", "greetings",
        "good morning", "good afternoon", "good evening",
    }
    THANKS = {"thanks", "thank you", "thanks!", "thank you!", "ty", "appreciate it"}
    FAREWELLS = {"bye", "goodbye", "see you", "see ya", "bye bye"}

    def _maybe_handle_smalltalk(self, question: str) -> Optional[str]:
        """Handles greetings/thanks/farewells directly, skipping retrieval
        for a faster, more natural response on simple chit-chat."""
        normalized = question.strip().lower().rstrip("!.?")

        if normalized in self.GREETINGS:
            return "Hello! I'm ready to help with this meeting. Ask me anything about what was discussed, decisions made, or action items."
        if normalized in self.THANKS:
            return "You're welcome! Let me know if you have any other questions about this meeting."
        if normalized in self.FAREWELLS:
            return "Goodbye! Feel free to come back anytime you have questions about this meeting."
        return None

    def answer(self, question: str, meeting_id: Optional[str] = None) -> QAResult:
        """
        Answers a question using retrieval-augmented generation.

        Args:
            question: the user's natural-language question.
            meeting_id: optionally restrict retrieval to one meeting.

        Returns:
            QAResult with the generated answer and supporting evidence.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        smalltalk_reply = self._maybe_handle_smalltalk(question)
        if smalltalk_reply is not None:
            return QAResult(answer=smalltalk_reply, evidence=[], question=question)

        # 1. Embed the question
        query_embedding = self.embed_model.encode(question).tolist()

        # 2. Retrieve relevant chunks
        results = self.vector_store.query(query_embedding, top_k=self.top_k, meeting_id=meeting_id)

        if not results:
            return QAResult(
                answer="I could not find this information in the meeting. (No relevant transcript data available.)",
                evidence=[],
                question=question,
            )

        # 3. Build context from retrieved chunks
        context_blocks = []
        evidence_list = []
        for r in results:
            meta = r["metadata"]
            timestamp = f"[{meta['start']:.0f}s-{meta['end']:.0f}s]"
            speaker = f" ({meta['speaker']})" if meta.get("speaker") else ""
            context_blocks.append(f"{timestamp}{speaker}: {r['text']}")
            evidence_list.append(
                Evidence(
                    text=r["text"],
                    start=meta["start"],
                    end=meta["end"],
                    speaker=meta.get("speaker") or None,
                    distance=r["distance"],
                )
            )

        context = "\n\n".join(context_blocks)

        # 4. Build the grounded prompt
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # 5. Generate the answer
        logger.info(f"Generating answer for question: '{question}'")
        try:
            raw_answer = self.llm.generate(prompt, temperature=0.2)
        except Exception as e:
            raw_answer = f"Answer generation failed: {e}"
            logger.error(raw_answer)

        return QAResult(answer=raw_answer, evidence=evidence_list, question=question)


if __name__ == "__main__":
    # Manual test runner.
    # Usage: python -m src.rag.qa_pipeline "your question here"
    import sys
    sys.path.insert(0, ".")

    from src.retrieval.vector_store import VectorStore
    from src.rag.llm_interface import get_llm

    if len(sys.argv) < 2:
        print('Usage: python src/rag/qa_pipeline.py "your question"')
        sys.exit(1)

    question = sys.argv[1]

    store = VectorStore()
    llm = get_llm(provider="ollama", model_name="llama3.2")
    pipeline = RAGPipeline(vector_store=store, llm=llm)

    result = pipeline.answer(question)

    print(f"\n=== Question ===\n{result.question}")
    print(f"\n=== Answer ===\n{result.answer}")
    print(f"\n=== Evidence ===")
    for e in result.evidence:
        print(f"\n[{e.start:.1f}s-{e.end:.1f}s] (distance: {e.distance:.4f})")
        print(e.text)









