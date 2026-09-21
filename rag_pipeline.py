"""
RAG-style advisory pipeline over the springshed management corpus.

Retrieval method: TF-IDF + cosine similarity over paragraph-level chunks.

Why TF-IDF instead of neural embeddings (Sentence-BERT, as used in the major
project): this environment's network access is restricted to package registries
(PyPI/npm/GitHub) and cannot reach model hosts like huggingface.co at runtime, so a
transformer embedding model cannot be downloaded here. TF-IDF is a legitimate,
well-understood retrieval method and works fully offline. The retrieval interface
below (`retrieve(query, k)`) is written so that swapping in a Sentence-BERT +
FAISS index (exactly the approach from the major project) is a drop-in replacement
if this is deployed somewhere with model access -- see
`embed_with_sentence_transformers_STUB` stub at the bottom for the swap point.

Answer generation: rather than calling an external LLM (which would need an API key
this project doesn't assume access to, and risks hallucinated advice on a topic with
real-world consequences), this module returns the most relevant retrieved passages
directly, lightly assembled into a grounded answer with source attribution. Every
sentence surfaced to the user is traceable to a specific corpus document -- there is
no generative "filling in the gaps." This is a deliberate responsible-AI choice for
a domain (water/land intervention advice) where confidently wrong generated text is
worse than a clearly-sourced retrieved passage.
"""

import glob
import os
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Resolve relative to this file's location.
# rag_pipeline.py and corpus/ are both in the repository root.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")


@dataclass
class Chunk:
    source: str
    title: str
    text: str


@dataclass
class RetrievedResult:
    chunk: Chunk
    score: float


def _load_corpus(corpus_dir: str = CORPUS_DIR) -> list[Chunk]:
    chunks = []

    # Find all .txt files inside the corpus directory
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):

        with open(path, encoding="utf-8") as f:
            text = f.read()

        title_match = re.match(r"Title:\s*(.+)", text)

        title = (
            title_match.group(1).strip()
            if title_match
            else os.path.basename(path)
        )

        body = (
            text.split("\n\n", 1)[-1]
            if "\n\n" in text
            else text
        )

        # Paragraph-level chunking
        paragraphs = [
            p.strip()
            for p in body.split("\n\n")
            if len(p.strip()) > 40
        ]

        for para in paragraphs:
            chunks.append(
                Chunk(
                    source=os.path.basename(path),
                    title=title,
                    text=para,
                )
            )

    return chunks


class SpringAdvisoryRAG:

    def __init__(self, corpus_dir: str = CORPUS_DIR):

        # Load corpus
        self.chunks = _load_corpus(corpus_dir)

        # Safety check so deployment gives a useful error
        # instead of sklearn's vague "empty vocabulary" error.
        if not self.chunks:
            raise ValueError(
                f"No corpus chunks found. "
                f"Checked corpus directory: {corpus_dir}"
            )

        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        # Create document-term matrix
        self._matrix = self.vectorizer.fit_transform(
            [c.text for c in self.chunks]
        )

    def retrieve(
        self,
        query: str,
        k: int = 4
    ) -> list[RetrievedResult]:

        # Convert query into TF-IDF vector
        q_vec = self.vectorizer.transform([query])

        # Calculate cosine similarity
        sims = cosine_similarity(
            q_vec,
            self._matrix
        ).flatten()

        # Get top-k results
        top_idx = sims.argsort()[::-1][:k]

        return [
            RetrievedResult(
                chunk=self.chunks[i],
                score=float(sims[i])
            )
            for i in top_idx
            if sims[i] > 0
        ]

    def answer(
        self,
        query: str,
        k: int = 3
    ) -> dict:

        results = self.retrieve(query, k=k)

        # No relevant information found
        if not results:
            return {
                "query": query,
                "answer": (
                    "I couldn't find grounded guidance on this in the "
                    "current corpus. Rather than guess, I'd recommend "
                    "consulting a local hydrogeologist or your district's "
                    "rural development / drinking water department."
                ),
                "sources": [],
            }

        sources = []
        answer_parts = []

        # Return retrieved passages directly
        for r in results:

            answer_parts.append(
                f'From "{r.chunk.title}": {r.chunk.text}'
            )

            sources.append(
                {
                    "title": r.chunk.title,
                    "file": r.chunk.source,
                    "relevance": round(r.score, 3),
                }
            )

        return {
            "query": query,
            "answer": "\n\n".join(answer_parts),
            "sources": sources,
        }


def embed_with_sentence_transformers_STUB():
    """
    Swap-in point for the major-project-style approach when deployed with
    model access.

    Possible future implementation:
    - Load all-MiniLM-L6-v2 using sentence-transformers
    - Generate embeddings for corpus chunks
    - Build a FAISS index
    - Encode the user query
    - Search the FAISS index

    This remains a stub because the current project is designed
    to work fully offline without external model downloads.
    """

    raise NotImplementedError(
        "Requires internet access to a model hub; "
        "see docstring."
    )


if __name__ == "__main__":

    rag = SpringAdvisoryRAG()

    print(
        f"Loaded {len(rag.chunks)} chunks "
        f"from corpus."
    )

    result = rag.answer(
        "How do we revive a spring that is drying up in our village?"
    )

    print(
        result["answer"][:500]
    )

    print(
        "\nSources:",
        result["sources"]
    )
