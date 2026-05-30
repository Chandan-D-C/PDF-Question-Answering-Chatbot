"""
PDF Loader Module
Handles PDF file reading, text extraction, and metadata extraction.
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import PyPDF2
import io

logger = logging.getLogger(__name__)


class PDFLoader:
    """Handles loading and text extraction from PDF files."""

    def __init__(self):
        self.loaded_files: List[str] = []

    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, Dict]:
        """
        Extract text and metadata from a single PDF file.

        Args:
            pdf_file: File-like object or path to PDF

        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        text = ""
        metadata = {}

        try:
            if isinstance(pdf_file, (str, Path)):
                with open(pdf_file, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text, metadata = self._process_reader(pdf_reader, str(pdf_file))
            else:
                # File-like object (e.g., from Streamlit uploader)
                pdf_bytes = pdf_file.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                text, metadata = self._process_reader(pdf_reader, pdf_file.name)

        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Corrupted PDF: {e}")
            raise ValueError(f"Could not read PDF file. It may be corrupted or encrypted: {e}")
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise

        return text, metadata

    def _process_reader(self, pdf_reader: PyPDF2.PdfReader, filename: str) -> Tuple[str, Dict]:
        """Process a PdfReader object to extract text and metadata."""
        pages_text = []
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                pages_text.append("")

        full_text = "\n\n".join(pages_text)

        # Extract metadata
        raw_meta = pdf_reader.metadata or {}
        metadata = {
            "filename": filename,
            "num_pages": num_pages,
            "title": raw_meta.get("/Title", Path(filename).stem),
            "author": raw_meta.get("/Author", "Unknown"),
            "subject": raw_meta.get("/Subject", ""),
            "creator": raw_meta.get("/Creator", ""),
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
        }

        self.loaded_files.append(filename)
        logger.info(f"Extracted {num_pages} pages from '{filename}' ({metadata['word_count']} words)")
        return full_text, metadata

    def extract_from_multiple_pdfs(self, pdf_files: List) -> Tuple[str, List[Dict]]:
        """
        Extract text from multiple PDF files.

        Args:
            pdf_files: List of file-like objects or paths

        Returns:
            Tuple of (combined_text, list_of_metadata)
        """
        all_texts = []
        all_metadata = []
        errors = []

        for pdf_file in pdf_files:
            try:
                text, metadata = self.extract_text_from_pdf(pdf_file)
                if text.strip():
                    all_texts.append(f"=== Document: {metadata['filename']} ===\n{text}")
                    all_metadata.append(metadata)
                else:
                    errors.append(f"'{metadata['filename']}' appears to be empty or image-based.")
            except ValueError as e:
                errors.append(str(e))
            except Exception as e:
                logger.error(f"Failed to process PDF: {e}")
                errors.append(f"Failed to process a PDF file: {e}")

        combined_text = "\n\n".join(all_texts)

        if errors:
            logger.warning(f"Encountered errors: {errors}")

        return combined_text, all_metadata, errors

    def get_loaded_files(self) -> List[str]:
        """Return list of successfully loaded file names."""
        return self.loaded_files
