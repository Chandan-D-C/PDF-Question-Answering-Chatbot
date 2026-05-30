"""
Text Splitter Module
Handles splitting extracted text into semantic chunks for embedding.
"""

import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits documents into optimally-sized chunks for retrieval."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

    def split_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        Split a single text string into chunks.

        Args:
            text: The text to split
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of LangChain Document objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to splitter")
            return []

        chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[metadata or {}]
        )

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def split_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[Document]:
        """
        Split multiple texts into chunks, preserving metadata.

        Args:
            texts: List of text strings
            metadatas: List of metadata dicts (one per text)

        Returns:
            List of LangChain Document objects with source metadata
        """
        if not texts:
            return []

        if metadatas is None:
            metadatas = [{} for _ in texts]

        all_chunks = []
        for idx, (text, meta) in enumerate(zip(texts, metadatas)):
            if not text.strip():
                continue
            chunks = self.splitter.create_documents(
                texts=[text],
                metadatas=[{**meta, "doc_index": idx}]
            )
            # Add chunk index within each document
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(texts)} documents")
        return all_chunks

    def split_combined_text(self, combined_text: str, pdf_metadata: List[Dict]) -> List[Document]:
        """
        Split combined text while preserving per-document metadata.

        Args:
            combined_text: Full extracted text from all PDFs
            pdf_metadata: List of metadata dicts from each PDF

        Returns:
            List of Document chunks with rich metadata
        """
        chunks = self.splitter.split_text(combined_text)
        documents = []

        for i, chunk in enumerate(chunks):
            # Try to identify which source document this chunk belongs to
            source_doc = self._identify_source(chunk, pdf_metadata)

            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "source": source_doc.get("filename", "Unknown"),
                    "title": source_doc.get("title", "Unknown"),
                    "char_count": len(chunk),
                    "word_count": len(chunk.split()),
                    **{k: v for k, v in source_doc.items() if k not in ["filename", "title"]},
                }
            )
            documents.append(doc)

        logger.info(f"Split combined text into {len(documents)} chunks")
        return documents

    def _identify_source(self, chunk: str, pdf_metadata: List[Dict]) -> Dict:
        """Heuristically identify which PDF a chunk came from."""
        if not pdf_metadata:
            return {}
        # Return first doc as default (works for single PDF scenarios)
        # Multi-PDF attribution is handled via document markers in combined text
        for meta in pdf_metadata:
            filename = meta.get("filename", "")
            if filename and filename.split(".")[0] in chunk[:200]:
                return meta
        return pdf_metadata[0] if pdf_metadata else {}

    def get_chunk_stats(self, chunks: List[Document]) -> Dict:
        """Return statistics about the chunks."""
        if not chunks:
            return {}
        lengths = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(lengths) / len(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
        }
