"""
Chatbot Module
Core RAG pipeline: retrieval + LLM generation with conversation memory.
"""

import json
import logging
import os
from typing import List, Dict, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .retriever import DocumentRetriever
from .prompt_template import (
    get_rag_prompt,
    get_summarization_prompt,
    get_keyword_prompt,
    format_chat_history,
)

logger = logging.getLogger(__name__)


class RAGChatbot:
    """End-to-end RAG chatbot with conversation memory and source attribution."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_provider: str = "google",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        """
        Initialize the chatbot.

        Args:
            retriever: DocumentRetriever instance
            llm_provider: "google" or "openai"
            temperature: LLM temperature (0=deterministic, 1=creative)
            max_tokens: Maximum tokens in LLM response
        """
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict] = []

        self.llm = self._init_llm()
        self.rag_prompt = get_rag_prompt()

    def _init_llm(self):
        """Initialize the LLM based on provider setting."""
        if self.llm_provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True,
            )
            logger.info("Initialized Gemini 2.5 Pro LLM")
            return llm

        elif self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            llm = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info("Initialized OpenAI GPT-4o LLM")
            return llm

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def ask(self, question: str) -> Dict:
        """
        Process a user question through the full RAG pipeline.

        Args:
            question: User's natural language question

        Returns:
            Dict with keys: answer, sources, confidence, context_used
        """
        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": [], "confidence": 0.0}

        # Step 1: Retrieve relevant chunks
        retrieved_results = self.retriever.retrieve(question)

        # Step 2: Format context and sources
        context = self.retriever.format_context(retrieved_results)
        sources = self.retriever.get_sources(retrieved_results)
        confidence = self.retriever.compute_confidence(retrieved_results)

        # Step 3: Format conversation history
        chat_history_str = format_chat_history(self.conversation_history)

        # Step 4: Build the prompt
        messages = self.rag_prompt.format_messages(
            context=context,
            chat_history=chat_history_str,
            question=question,
        )

        # Step 5: Call the LLM
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = f"Sorry, I encountered an error while generating a response: {str(e)}"
            confidence = 0.0

        # Step 6: Update conversation memory
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Keep memory bounded (last 20 messages = 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
        }

    def summarize_documents(self, combined_text: str) -> str:
        """
        Generate a summary of all uploaded documents.

        Args:
            combined_text: Full text from all PDFs

        Returns:
            Structured document summary
        """
        prompt = get_summarization_prompt()
        # Truncate to avoid token limits
        truncated = combined_text[:8000] + "\n[... document truncated for summary ...]" if len(combined_text) > 8000 else combined_text
        formatted = prompt.format(content=truncated)

        try:
            response = self.llm.invoke([HumanMessage(content=formatted)])
            return response.content
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Could not generate summary: {e}"

    def extract_keywords(self, text: str) -> Dict:
        """
        Extract keywords and key phrases from document text.

        Args:
            text: Document text

        Returns:
            Dict with keywords, key_phrases, topics
        """
        prompt = get_keyword_prompt()
        truncated = text[:4000]
        formatted = prompt.format(text=truncated)

        try:
            response = self.llm.invoke([HumanMessage(content=formatted)])
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Could not parse keyword JSON, returning raw text")
            return {"keywords": [], "key_phrases": [], "topics": [], "raw": response.content}
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {"keywords": [], "key_phrases": [], "topics": []}

    def clear_memory(self):
        """Reset conversation history."""
        self.conversation_history = []
        logger.info("Conversation memory cleared")

    def get_history(self) -> List[Dict]:
        """Return the full conversation history."""
        return self.conversation_history

    def export_chat_history(self) -> str:
        """Export chat history as formatted text."""
        if not self.conversation_history:
            return "No conversation history."

        lines = ["=== Chat History Export ===\n"]
        for i, msg in enumerate(self.conversation_history):
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}:\n{msg['content']}\n")
            if i % 2 == 1:
                lines.append("-" * 50 + "\n")

        return "\n".join(lines)
