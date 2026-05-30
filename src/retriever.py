"""
Retriever Module
Handles document retrieval and context assembly for RAG pipeline.
"""

import logging
from typing import List, Dict, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieves and formats relevant document chunks for LLM context."""

    def __init__(self, vector_store_manager, top_k: int = 4, use_hybrid: bool = True):
        """
        Initialize retriever.

        Args:
            vector_store_manager: VectorStoreManager instance
            top_k: Number of chunks to retrieve
            use_hybrid: Whether to use hybrid (semantic + keyword) search
        """
        self.vsm = vector_store_manager
        self.top_k = top_k
        self.use_hybrid = use_hybrid

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: User's question

        Returns:
            List of (Document, relevance_score) tuples
        """
        if self.use_hybrid:
            results = self.vsm.hybrid_search(query, k=self.top_k)
        else:
            results = self.vsm.similarity_search(query, k=self.top_k)

        return results

    def format_context(self, results: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.

        Args:
            results: List of (Document, score) tuples

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found in the uploaded documents."

        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Context {i} | Source: {source} | Relevance: {score:.2f}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self, results: List[Tuple[Document, float]]) -> List[Dict]:
        """
        Extract source attribution information from results.

        Args:
            results: Retrieved documents with scores

        Returns:
            List of source info dicts
        """
        sources = []
        seen = set()

        for doc, score in results:
            meta = doc.metadata
            source = meta.get("source", "Unknown")
            chunk_id = meta.get("chunk_id", 0)
            key = f"{source}_{chunk_id}"

            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source,
                    "title": meta.get("title", source),
                    "chunk_id": chunk_id,
                    "relevance_score": round(score, 3),
                    "excerpt": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "word_count": meta.get("word_count", len(doc.page_content.split())),
                })

        return sorted(sources, key=lambda x: x["relevance_score"], reverse=True)

    def compute_confidence(self, results: List[Tuple[Document, float]]) -> float:
        """
        Compute an overall confidence score for the retrieved context.

        Args:
            results: Retrieved documents with scores

        Returns:
            Confidence score between 0 and 1
        """
        if not results:
            return 0.0

        scores = [score for _, score in results]
        # Weighted average: top result has higher weight
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        total_weight = sum(weights)
        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return round(min(1.0, max(0.0, weighted_avg)), 3)
