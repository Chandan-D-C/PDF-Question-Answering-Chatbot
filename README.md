# 📚 PDF RAG Chatbot — Intelligent Document Question Answering

_An AI-powered Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and interact with them using natural language through Gemini/OpenAI, LangChain, and FAISS._

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#business-problem">Business Problem</a>
- <a href="#features">Features</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#rag-workflow">RAG Workflow</a>
- <a href="#application-interface">Application Interface</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#future-implementations">Future Implementations</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

PDF RAG Chatbot is an intelligent document analysis platform that allows users to upload one or multiple PDF files and ask questions in natural language. The application leverages Retrieval-Augmented Generation (RAG) to retrieve relevant document content and generate accurate responses using Large Language Models such as Gemini 2.5 Pro or OpenAI GPT.

The system combines semantic search, vector embeddings, and conversational AI to provide contextual answers grounded in uploaded documents.

---

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

Organizations and individuals often work with lengthy documents, research papers, reports, and manuals that are difficult to search manually.

This project aims to:

- Enable intelligent document search using AI.
- Reduce time spent reading large PDFs.
- Improve information retrieval accuracy through semantic search.
- Generate contextual answers directly from uploaded documents.
- Provide a conversational interface for document exploration.

---

<h2><a class="anchor" id="features"></a>Features</h2>

### 📄 Document Processing
- Upload multiple PDF documents
- Automatic text extraction
- Metadata generation
- Document chunking for efficient retrieval

### 🔍 Intelligent Retrieval
- Semantic similarity search
- Hybrid retrieval (semantic + keyword search)
- FAISS vector database indexing
- Top-K document retrieval

### 🤖 AI-Powered Question Answering
- Gemini 2.5 Pro support
- OpenAI GPT support
- Context-aware answer generation
- Confidence score estimation
- Source citation display

### 📊 Document Analytics
- Document summarization
- Keyword extraction
- Topic identification
- Document statistics dashboard

### 💬 Conversational Interface
- Multi-turn chat memory
- Dark & Light mode support
- Chat history export (TXT/JSON)

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- **Frontend/UI**: Streamlit
- **Framework**: LangChain
- **Vector Database**: FAISS
- **LLMs**: Google Gemini 2.5 Pro, OpenAI GPT
- **PDF Processing**: PyPDF2
- **Embeddings**: Google Generative AI Embeddings
- **Environment Management**: python-dotenv
- **Programming Language**: Python

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```text
pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
│
├── src/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── chatbot.py
│
├── data/
│   └── pdf_files/
│
├── vectorstore/
│   └── faiss_index/
│
└── README.md
```

---

<h2><a class="anchor" id="rag-workflow"></a>RAG Workflow</h2>

The chatbot follows a Retrieval-Augmented Generation pipeline:

```text
PDF Upload
     │
     ▼
Text Extraction
     │
     ▼
Chunking Documents
     │
     ▼
Generate Embeddings
     │
     ▼
Store in FAISS
     │
     ▼
User Question
     │
     ▼
Retrieve Relevant Chunks
     │
     ▼
Gemini / OpenAI LLM
     │
     ▼
Context-Aware Answer
```

### Processing Steps

1. Extract text from uploaded PDFs.
2. Split documents into semantic chunks.
3. Generate vector embeddings.
4. Store embeddings inside FAISS.
5. Retrieve relevant chunks based on user queries.
6. Send retrieved context to LLM.
7. Generate grounded responses with source references.

---

<h2><a class="anchor" id="application-interface"></a>Application Interface</h2>

### Main Features

- 📚 Upload Multiple PDFs
- 🔑 API Key Management
- 🔍 Hybrid Search Configuration
- 📊 Document Statistics Dashboard
- 💬 Interactive Chat Window
- 📝 AI Document Summary
- 🏷️ Keyword Extraction
- 📥 Chat Export Functionality

### Dashboard Highlights

- Dark/Light Mode Toggle
- Chunk & Document Metrics
- Source Relevance Scores
- Confidence-Based Responses
- Responsive Streamlit Interface

> Add screenshots here after deployment.

```markdown
![Dashboard Screenshot](images/dashboard.png)

![Chat Interface](images/chatbot.png)
```

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git

cd pdf-rag-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here

OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run Application

```bash
streamlit run app.py
```

### 6. Open Browser

```text
http://localhost:8501
```

---

<h2><a class="anchor" id="future-implementations"></a>Future Implementations</h2>

- OCR support for scanned PDFs
- Multi-format document support (DOCX, PPTX, TXT)
- Persistent vector database storage
- User authentication system
- Cloud deployment on AWS/Azure/GCP
- Citation highlighting inside PDF pages
- Multi-language document understanding
- Advanced document comparison tools

---

## 👨‍💻 Author

**Chandan**

AI/ML & Generative AI Developer

- LinkedIn: https://www.linkedin.com/in/chandan-d-c-803934272/
- GitHub: https://github.com/Chandan-D-C


---

