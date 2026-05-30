"""
PDF RAG Chatbot - Main Streamlit Application
Intelligent PDF Question Answering using RAG, LangChain, FAISS, and Gemini
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# ─── Environment & Logging Setup ─────────────────────────────────────────────
load_dotenv()

# Clear out placeholder keys from environment variables
for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY"]:
    val = os.getenv(key)
    if val and ("your_" in val or "_here" in val or val.strip() == ""):
        os.environ[key] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ─── Add src to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_loader import PDFLoader
from src.text_splitter import TextChunker
from src.embeddings import get_embeddings
from src.vector_store import VectorStoreManager
from src.retriever import DocumentRetriever
from src.chatbot import RAGChatbot

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/pdf-rag-chatbot",
        "About": "Intelligent PDF Q&A powered by RAG + Gemini",
    },
)

# ─── CSS Styling (Light + Dark Mode) ─────────────────────────────────────────
def load_css(dark_mode: bool = False):
    base_bg = "#0e1117" if dark_mode else "#f8f9fa"
    sidebar_bg = "#1a1d24" if dark_mode else "#ffffff"
    card_bg = "#1e2130" if dark_mode else "#ffffff"
    text_color = "#e8eaf6" if dark_mode else "#212529"
    secondary_text = "#9fa8da" if dark_mode else "#6c757d"
    accent = "#7c4dff"
    accent_light = "#b39ddb"
    border_color = "#2d3250" if dark_mode else "#dee2e6"
    user_msg_bg = "#283593" if dark_mode else "#e8eaf6"
    bot_msg_bg = "#1a237e" if dark_mode else "#f3e5f5"
    input_bg = "#1e2130" if dark_mode else "#ffffff"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {{ font-family: 'Inter', sans-serif !important; box-sizing: border-box; }}

        .stApp {{ background-color: {base_bg}; color: {text_color}; }}
        [data-testid="stSidebar"] {{ background-color: {sidebar_bg}; border-right: 1px solid {border_color}; }}
        [data-testid="stSidebar"] * {{ color: {text_color}; }}

        /* Header */
        .app-header {{
            background: linear-gradient(135deg, {accent} 0%, #e91e63 100%);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(124,77,255,0.3);
            text-align: center;
        }}
        .app-header h1 {{ color: white; font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }}
        .app-header p {{ color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 0.5rem 0 0; }}

        /* Cards */
        .card {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}

        /* Status badges */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-success {{ background: #1b5e20; color: #a5d6a7; }}
        .badge-info {{ background: #0d47a1; color: #90caf9; }}
        .badge-warning {{ background: #e65100; color: #ffcc80; }}
        .badge-purple {{ background: #4a148c; color: {accent_light}; }}

        /* Chat messages */
        .chat-user {{
            background: {user_msg_bg};
            border: 1px solid {border_color};
            border-radius: 12px 12px 2px 12px;
            padding: 1rem 1.25rem;
            margin: 0.5rem 0;
            max-width: 85%;
            margin-left: auto;
            color: {text_color};
        }}
        .chat-bot {{
            background: {bot_msg_bg};
            border: 1px solid {border_color};
            border-radius: 2px 12px 12px 12px;
            padding: 1rem 1.25rem;
            margin: 0.5rem 0;
            max-width: 85%;
            color: {text_color};
        }}
        .chat-label {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: {secondary_text};
            margin-bottom: 0.4rem;
        }}

        /* Confidence meter */
        .confidence-bar {{
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(90deg, #f44336, #ff9800, #4caf50);
            margin-top: 0.5rem;
        }}

        /* Source card */
        .source-card {{
            background: {card_bg};
            border-left: 3px solid {accent};
            border-radius: 0 8px 8px 0;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            color: {text_color};
        }}

        /* Sidebar sections */
        .sidebar-section {{
            background: {card_bg};
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid {border_color};
        }}
        .sidebar-section h4 {{
            color: {accent_light};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 0 0 0.75rem;
            font-weight: 600;
        }}

        /* Styledbuttons */
        .stButton > button {{
            background: linear-gradient(135deg, {accent}, #e91e63);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(124,77,255,0.3);
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(124,77,255,0.4);
        }}

        /* File uploader */
        [data-testid="stFileUploader"] {{
            border: 2px dashed {border_color};
            border-radius: 10px;
            padding: 0.5rem;
        }}

        /* Metrics */
        [data-testid="metric-container"] {{
            background: {card_bg};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 0.75rem !important;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background: {card_bg};
            border-radius: 8px;
            color: {text_color} !important;
        }}

        /* Scrollable chat */
        .chat-container {{
            max-height: 520px;
            overflow-y: auto;
            padding: 0.5rem;
            scrollbar-width: thin;
            scrollbar-color: {accent} transparent;
        }}

        /* Tag pills */
        .tag {{
            display: inline-block;
            background: {card_bg};
            border: 1px solid {accent};
            color: {accent_light};
            border-radius: 20px;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            margin: 0.15rem;
        }}

        /* Divider */
        hr {{ border-color: {border_color}; margin: 1rem 0; }}

        /* Streamlit overrides */
        .stTextInput > div > div > input, .stTextArea textarea {{
            background: {input_bg} !important;
            color: {text_color} !important;
            border-color: {border_color} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox > div > div {{
            background: {input_bg} !important;
            color: {text_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chatbot": None,
        "vector_store_manager": None,
        "pdf_metadata": [],
        "combined_text": "",
        "chat_history": [],
        "processing_done": False,
        "dark_mode": True,
        "llm_provider": "google",
        "top_k": 4,
        "use_hybrid": True,
        "document_summary": "",
        "extracted_keywords": {},
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─── Processing Pipeline ──────────────────────────────────────────────────────
def process_pdfs(uploaded_files, provider: str, top_k: int, use_hybrid: bool):
    """Full RAG processing pipeline."""
    progress = st.progress(0, text="Starting...")
    status = st.empty()

    try:
        # Step 1: Extract text
        status.info("📄 Extracting text from PDFs...")
        progress.progress(15, text="Extracting text...")

        loader = PDFLoader()
        combined_text, pdf_metadata, errors = loader.extract_from_multiple_pdfs(uploaded_files)

        if errors:
            for err in errors:
                st.warning(f"⚠️ {err}")

        if not combined_text.strip():
            st.error("❌ No text could be extracted from the uploaded files.")
            return False

        # Step 2: Split into chunks
        status.info("✂️ Splitting documents into semantic chunks...")
        progress.progress(35, text="Chunking...")

        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        documents = chunker.split_combined_text(combined_text, pdf_metadata)
        st.session_state.total_chunks = len(documents)

        if not documents:
            st.error("❌ No chunks created. Documents may be empty.")
            return False

        # Step 3: Generate embeddings
        status.info("🧠 Generating embeddings...")
        progress.progress(55, text="Embedding...")

        embeddings = get_embeddings(provider=provider)

        # Step 4: Build vector store
        status.info("🗄️ Building FAISS vector store...")
        progress.progress(75, text="Indexing...")

        vsm = VectorStoreManager()
        vsm.create_vector_store(documents, embeddings)

        # Step 5: Init retriever and chatbot
        status.info("🤖 Initializing chatbot...")
        progress.progress(90, text="Initializing AI...")

        retriever = DocumentRetriever(vsm, top_k=top_k, use_hybrid=use_hybrid)
        chatbot = RAGChatbot(retriever, llm_provider=provider)

        # Save to session
        st.session_state.chatbot = chatbot
        st.session_state.vector_store_manager = vsm
        st.session_state.pdf_metadata = pdf_metadata
        st.session_state.combined_text = combined_text
        st.session_state.processing_done = True
        st.session_state.chat_history = []
        st.session_state.document_summary = ""
        st.session_state.extracted_keywords = {}

        progress.progress(100, text="Done!")
        status.success(f"✅ Processed {len(pdf_metadata)} PDF(s) → {len(documents)} chunks indexed!")
        return True

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        progress.empty()
        st.error(f"❌ Processing failed: {str(e)}")
        return False


# ─── UI Components ────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="app-header">
        <h1>📚 Intelligent PDF Chatbot</h1>
        <p>Retrieval-Augmented Generation · LangChain · FAISS · Gemini 2.5 Pro</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h4>⚙️ Configuration</h4>', unsafe_allow_html=True)

        # Dark mode toggle
        dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.divider()

        # LLM provider
        provider = st.selectbox(
            "🤖 LLM Provider",
            options=["google", "openai"],
            index=0 if st.session_state.llm_provider == "google" else 1,
            format_func=lambda x: "Gemini 2.5 Pro" if x == "google" else "OpenAI GPT-4o",
        )
        st.session_state.llm_provider = provider

        # Retrieval settings
        st.session_state.top_k = st.slider("🔍 Top-K Chunks", 2, 8, st.session_state.top_k)
        st.session_state.use_hybrid = st.checkbox("⚡ Hybrid Search", value=st.session_state.use_hybrid,
                                                    help="Combines semantic + keyword search for better results")

        st.markdown('</div>', unsafe_allow_html=True)

        # API Credentials section
        st.markdown('<div class="sidebar-section"><h4>🔑 API Credentials</h4>', unsafe_allow_html=True)
        env_google_key = os.getenv("GOOGLE_API_KEY", "")
        if env_google_key == "your_google_api_key_here":
            env_google_key = ""
        env_openai_key = os.getenv("OPENAI_API_KEY", "")
        if env_openai_key == "your_openai_api_key_here":
            env_openai_key = ""
            
        google_api_key = st.text_input(
            "Google API Key",
            value=st.session_state.get("google_api_key", env_google_key),
            type="password",
            help="Required for Gemini. Get from https://aistudio.google.com",
        )
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=st.session_state.get("openai_api_key", env_openai_key),
            type="password",
            help="Required only if using OpenAI. Get from https://platform.openai.com",
        )
        
        st.session_state["google_api_key"] = google_api_key
        st.session_state["openai_api_key"] = openai_api_key
        
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        st.markdown('</div>', unsafe_allow_html=True)

        # Upload section
        st.markdown('<div class="sidebar-section"><h4>📤 Upload PDFs</h4>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            for f in uploaded_files:
                size_kb = len(f.getvalue()) / 1024
                st.markdown(f'<div style="font-size:0.8rem; padding:0.3rem 0;">📄 <b>{f.name}</b> ({size_kb:.1f} KB)</div>', unsafe_allow_html=True)

            # Check if required API key is set
            provider = st.session_state.llm_provider
            api_key_set = False
            if provider == "google" and os.environ.get("GOOGLE_API_KEY"):
                api_key_set = True
            elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
                api_key_set = True

            if not api_key_set:
                st.warning(f"⚠️ Please enter your {'Google' if provider == 'google' else 'OpenAI'} API Key in the credentials section above to proceed.")
            else:
                if st.button("🚀 Process Documents", use_container_width=True):
                    process_pdfs(
                        uploaded_files,
                        provider=provider,
                        top_k=st.session_state.top_k,
                        use_hybrid=st.session_state.use_hybrid,
                    )

        st.markdown('</div>', unsafe_allow_html=True)

        # Stats
        if st.session_state.processing_done:
            st.markdown('<div class="sidebar-section"><h4>📊 Index Statistics</h4>', unsafe_allow_html=True)

            docs = st.session_state.pdf_metadata
            st.metric("📁 Documents", len(docs))
            st.metric("🧩 Chunks", st.session_state.total_chunks)
            total_words = sum(m.get("word_count", 0) for m in docs)
            st.metric("📝 Total Words", f"{total_words:,}")
            st.metric("🔍 Search Mode", "Hybrid" if st.session_state.use_hybrid else "Semantic")

            st.markdown('</div>', unsafe_allow_html=True)

            # Document info
            st.markdown('<div class="sidebar-section"><h4>📄 Loaded Documents</h4>', unsafe_allow_html=True)
            for meta in docs:
                st.markdown(f"""
                <div style="font-size:0.8rem; margin-bottom:0.4rem;">
                    <b>{meta.get('title', meta.get('filename', 'Unknown'))}</b><br>
                    <span style="opacity:0.7;">{meta.get('num_pages', '?')} pages · {meta.get('word_count', 0):,} words</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Clear controls
        if st.session_state.processing_done:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    if st.session_state.chatbot:
                        st.session_state.chatbot.clear_memory()
                    st.rerun()
            with col2:
                if st.button("🔄 Reset All", use_container_width=True):
                    for key in ["chatbot", "vector_store_manager", "pdf_metadata",
                                "combined_text", "chat_history", "processing_done",
                                "document_summary", "extracted_keywords", "total_chunks"]:
                        st.session_state[key] = None if key in ["chatbot", "vector_store_manager"] else (
                            [] if "history" in key or "metadata" in key else
                            "" if key in ["combined_text", "document_summary"] else
                            {} if key == "extracted_keywords" else
                            False if key == "processing_done" else 0
                        )
                    st.rerun()

        return uploaded_files


def render_chat_interface():
    """Render the main chat interface."""
    if not st.session_state.processing_done:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
            <h3 style="margin-bottom: 0.5rem;">Upload PDFs to Get Started</h3>
            <p style="opacity: 0.7;">Use the sidebar to upload your PDF documents, then ask questions in natural language.</p>
            <br>
            <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
                <span class="tag">🔍 Semantic Search</span>
                <span class="tag">🤖 Gemini 2.5 Pro</span>
                <span class="tag">📊 FAISS Vector DB</span>
                <span class="tag">🔗 LangChain</span>
                <span class="tag">💬 Chat Memory</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label">You</div>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                conf = msg.get("confidence", 0)
                conf_pct = int(conf * 100)
                conf_color = "#4caf50" if conf > 0.7 else ("#ff9800" if conf > 0.4 else "#f44336")
                st.markdown(f"""
                <div class="chat-bot">
                    <div class="chat-label">Assistant · Confidence: <span style="color:{conf_color}">{conf_pct}%</span></div>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)

                # Show sources inline
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} Source(s) Used", expanded=False):
                        for src in msg["sources"]:
                            score_pct = int(src["relevance_score"] * 100)
                            st.markdown(f"""
                            <div class="source-card">
                                <b>📄 {src['title']}</b> &nbsp;
                                <span class="badge badge-info">{score_pct}% match</span><br>
                                <small style="opacity:0.8;">{src['excerpt']}</small>
                            </div>
                            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    st.divider()

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(3)
        suggestions = [
            "Summarize the main topics",
            "What are the key findings?",
            "List the important dates mentioned",
        ]
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(f"💬 {suggestion}", use_container_width=True, key=f"sug_{suggestion}"):
                    _process_question(suggestion)

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask a question about your documents...",
                placeholder="e.g. What is the main conclusion of the paper?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

        if submitted and user_input.strip():
            _process_question(user_input.strip())


def _process_question(question: str):
    """Process a user question and update chat history."""
    chatbot = st.session_state.chatbot
    if not chatbot:
        st.error("Chatbot not initialized. Please upload and process PDFs first.")
        return

    with st.spinner("🔍 Retrieving context and generating answer..."):
        result = chatbot.ask(question)

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
    })
    st.rerun()


def render_tools_panel():
    """Render document analysis tools panel."""
    if not st.session_state.processing_done:
        return

    st.subheader("🛠️ Document Analysis Tools")

    tab1, tab2, tab3 = st.tabs(["📋 Summary", "🏷️ Keywords", "💾 Export"])

    with tab1:
        if st.button("Generate Document Summary", use_container_width=True):
            with st.spinner("Summarizing documents..."):
                summary = st.session_state.chatbot.summarize_documents(
                    st.session_state.combined_text
                )
                st.session_state.document_summary = summary
        if st.session_state.document_summary:
            st.markdown(st.session_state.document_summary)

    with tab2:
        if st.button("Extract Keywords & Topics", use_container_width=True):
            with st.spinner("Extracting keywords..."):
                kw_data = st.session_state.chatbot.extract_keywords(
                    st.session_state.combined_text[:4000]
                )
                st.session_state.extracted_keywords = kw_data

        if st.session_state.extracted_keywords:
            kw = st.session_state.extracted_keywords
            if kw.get("keywords"):
                st.markdown("**Keywords:**")
                st.markdown(" ".join([f'<span class="tag">{k}</span>' for k in kw["keywords"]]), unsafe_allow_html=True)
            if kw.get("key_phrases"):
                st.markdown("**Key Phrases:**")
                st.markdown(" ".join([f'<span class="tag">{p}</span>' for p in kw["key_phrases"]]), unsafe_allow_html=True)
            if kw.get("topics"):
                st.markdown("**Main Topics:**")
                st.markdown(" ".join([f'<span class="tag">📌 {t}</span>' for t in kw["topics"]]), unsafe_allow_html=True)

    with tab3:
        if st.session_state.chat_history:
            export_text = st.session_state.chatbot.export_chat_history()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Download Chat History (.txt)",
                data=export_text,
                file_name=f"chat_history_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # Also export as JSON
            export_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="⬇️ Download Chat History (.json)",
                data=export_json,
                file_name=f"chat_history_{timestamp}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("No chat history to export yet.")

        # PDF metadata export
        if st.session_state.pdf_metadata:
            meta_json = json.dumps(st.session_state.pdf_metadata, indent=2)
            st.download_button(
                label="⬇️ Download Document Metadata (.json)",
                data=meta_json,
                file_name="document_metadata.json",
                mime="application/json",
                use_container_width=True,
            )


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session_state()
    load_css(dark_mode=st.session_state.dark_mode)

    render_header()

    # Sidebar (returns uploaded files)
    render_sidebar()

    # Main content area
    col_chat, col_tools = st.columns([3, 1])

    with col_chat:
        render_chat_interface()

    with col_tools:
        render_tools_panel()


if __name__ == "__main__":
    main()
