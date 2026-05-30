"""
Embeddings Module
Handles embedding generation using Google Generative AI or OpenAI.
"""

import logging
import os
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_embeddings(provider: str = "google"):
    """
    Factory function to get the appropriate embeddings model.

    Args:
        provider: "google" or "openai"

    Returns:
        Embeddings instance
    """
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        logger.info("Initialized Google Generative AI Embeddings")
        return embeddings

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        logger.info("Initialized OpenAI Embeddings")
        return embeddings

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Use 'google' or 'openai'.")
