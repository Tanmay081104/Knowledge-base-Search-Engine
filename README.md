# Knowledge-Based Search Engine

A Retrieval-Augmented Generation (RAG) system that provides intelligent document-based question answering using vector search and large language models.

## Features

- **Document Processing**: Support for PDF, DOCX, TXT, and MD files
- **Semantic Search**: Vector-based document retrieval using embeddings
- **Multi-LLM Support**: Integrates with multiple language models
- **Web Interface**: Clean, responsive frontend for document upload and querying
- **RESTful API**: FastAPI-based backend with comprehensive endpoints

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   └── config.py        # Configuration management
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── services/
│       ├── rag_service.py         # RAG orchestration
│       ├── vector_store.py        # Vector database integration
│       └── document_processor.py  # Document parsing
├── frontend/
│   └── templates/
│       └── index_clean.html   # Web interface
├── documents/                 # Sample documents
├── requirements.txt           # Project dependencies
└── simple_main.py            # Simplified version
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Tanmay081104/Knowledge-based-search-engine.git
cd Knowledge-based-search-engine
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with your API keys:

```env
# LLM API Keys (add at least one)
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROQ_API_KEY=your_groq_api_key

# Default LLM provider
DEFAULT_LLM=google
```

## Usage

### Option 1: Full Application
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Option 2: Simplified Version
```bash
python simple_main.py
```

Open your browser and navigate to `http://127.0.0.1:8000`

**Basic workflow:**
1. Upload documents (PDF, TXT, MD)
2. Ask questions about your documents
3. Get AI-powered answers with source references

## API Endpoints

- `POST /upload` - Upload and process documents
- `POST /query` - Submit questions and get AI answers
- `GET /health` - System health check
- `GET /docs` - Interactive API documentation

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Vector Database**: ChromaDB with sentence-transformers
- **LLM Integration**: OpenAI, Anthropic, Google Gemini, Groq
- **Document Processing**: PyPDF2, python-docx
- **Frontend**: HTML5, CSS3, JavaScript

## Requirements

- Python 3.8 or higher
- At least one LLM API key (OpenAI, Google, Anthropic, or Groq)
- Internet connection for API calls

To view the demo watch : https://drive.google.com/file/d/1Fg09aAp3H9cXPeYZLJLryWVl-ck4w8-i/view?usp=sharing
