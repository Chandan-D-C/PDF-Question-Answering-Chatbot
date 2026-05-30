"""
Prompt Template Module
Defines and manages prompt templates for the RAG pipeline.
"""

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# ─── Core RAG Prompt ────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an intelligent document assistant with expertise in analyzing and explaining content from uploaded PDF documents.

INSTRUCTIONS:
1. Answer questions ONLY based on the provided context from the documents.
2. If the answer cannot be found in the context, respond with: "I could not find this information in the uploaded documents."
3. Be concise, accurate, and well-structured in your responses.
4. When citing information, reference the source document when possible.
5. If the question is ambiguous, address the most likely interpretation.
6. Use bullet points or numbered lists when explaining multiple points.
7. Never fabricate information or make assumptions beyond the provided context.

CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}
"""

RAG_HUMAN_PROMPT = """Question: {question}

Please provide a comprehensive answer based on the document context above."""


# ─── Summarization Prompt ────────────────────────────────────────────────────

SUMMARIZATION_PROMPT = """You are a professional document summarizer.

Analyze the following document content and provide:
1. **Executive Summary** (2-3 sentences)
2. **Key Topics** (bullet points)
3. **Main Findings or Points** (numbered list)
4. **Notable Details** (if any)

DOCUMENT CONTENT:
{content}

Provide a structured, professional summary."""


# ─── Keyword Extraction Prompt ────────────────────────────────────────────────

KEYWORD_EXTRACTION_PROMPT = """Extract the most important keywords and key phrases from the following text.

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{{
  "keywords": ["keyword1", "keyword2", ...],
  "key_phrases": ["phrase1", "phrase2", ...],
  "topics": ["topic1", "topic2", ...]
}}

TEXT:
{text}"""


# ─── Prompt Factory ──────────────────────────────────────────────────────────

def get_rag_prompt() -> ChatPromptTemplate:
    """Return the main RAG chat prompt template."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(RAG_HUMAN_PROMPT),
    ])


def get_summarization_prompt() -> PromptTemplate:
    """Return the document summarization prompt."""
    return PromptTemplate(
        input_variables=["content"],
        template=SUMMARIZATION_PROMPT,
    )


def get_keyword_prompt() -> PromptTemplate:
    """Return the keyword extraction prompt."""
    return PromptTemplate(
        input_variables=["text"],
        template=KEYWORD_EXTRACTION_PROMPT,
    )


def format_chat_history(history: list) -> str:
    """Format conversation history for inclusion in prompt."""
    if not history:
        return "No previous conversation."

    formatted = []
    for msg in history[-6:]:  # Keep last 6 exchanges to stay within context limits
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)
