"""
Vector Store Module
Manages FAISS vector database: creation, persistence, and retrieval.
"""

import logging
import os
import pickle
from typing import List, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

VECTORSTORE_PATH = "vectorstore/faiss_index"


class VectorStoreManager:
    """Manages the FAISS vector store for document embeddings."""

    def __init__(self, store_path: str = VECTORSTORE_PATH):
        self.store_path = store_path
        self.vector_store: Optional[FAISS] = None
        os.makedirs(os.path.dirname(store_path), exist_ok=True)

    def create_vector_store(self, documents: List[Document], embeddings) -> FAISS:
        """
        Create a new FAISS vector store from documents.

        Args:
            documents: List of Document chunks
            embeddings: Embeddings model instance

        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")

        logger.info(f"Creating FAISS vector store with {len(documents)} chunks...")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )

        self.save_vector_store()
        logger.info("Vector store created and saved successfully")
        return self.vector_store

    def save_vector_store(self):
        """Persist the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        self.vector_store.save_local(self.store_path)
        logger.info(f"Vector store saved to '{self.store_path}'")

    def load_vector_store(self, embeddings) -> Optional[FAISS]:
        """
        Load an existing FAISS vector store from disk.

        Args:
            embeddings: Embeddings model (must match what was used to create it)

        Returns:
            FAISS vector store or None if not found
        """
        index_file = Path(self.store_path) / "index.faiss"
        if not index_file.exists():
            logger.info("No existing vector store found")
            return None

        try:
            self.vector_store = FAISS.load_local(
                self.store_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Loaded vector store from '{self.store_path}'")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None

    def add_documents(self, documents: List[Document], embeddings):
        """Add new documents to an existing vector store."""
        if self.vector_store is None:
            self.create_vector_store(documents, embeddings)
        else:
            self.vector_store.add_documents(documents)
            self.save_vector_store()
            logger.info(f"Added {len(documents)} documents to existing vector store")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: User query string
            k: Number of top results to return
            score_threshold: Minimum similarity score

        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please upload and process PDFs first.")

        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k,
        )

        # Filter by threshold
        if score_threshold > 0:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
        return results

    def hybrid_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining semantic similarity and keyword matching.

        Args:
            query: User query
            k: Number of results

        Returns:
            Ranked list of (Document, score) tuples
        """
        # Semantic search
        semantic_results = self.similarity_search(query, k=k * 2)

        # Keyword scoring boost
        query_keywords = set(query.lower().split())
        reranked = []

        for doc, score in semantic_results:
            content_lower = doc.page_content.lower()
            keyword_hits = sum(1 for kw in query_keywords if kw in content_lower)
            keyword_boost = keyword_hits * 0.05
            combined_score = min(1.0, score + keyword_boost)
            reranked.append((doc, combined_score))

        # Sort by combined score and return top-k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]

    def vector_store_exists(self) -> bool:
        """Check if a saved vector store exists on disk."""
        return (Path(self.store_path) / "index.faiss").exists()

    def clear_vector_store(self):
        """Remove the persisted vector store."""
        import shutil
        if Path(self.store_path).exists():
            shutil.rmtree(self.store_path)
            os.makedirs(self.store_path, exist_ok=True)
        self.vector_store = None
        logger.info("Vector store cleared")
