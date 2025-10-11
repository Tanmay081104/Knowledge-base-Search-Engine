# Knowledge Base Search Engine

A powerful RAG (Retrieval-Augmented Generation) system built with **Groq API integration** for lightning-fast AI responses. Upload documents, ask questions, and get intelligent answers synthesized from your knowledge base.

## 🎯 Features

### Core RAG System
- 📄 **Multi-format document processing** (PDF, TXT, MD, DOCX, XLSX, JSON, Images with OCR)
- 🔍 **Vector similarity search** with ChromaDB and sentence transformers
- 🤖 **Multi-LLM support** with **Groq as primary provider** (fastest inference)
- ⚡ **Real-time answer synthesis** using retrieved context

### Advanced Capabilities
- 🌐 **Interactive web interface** with document upload and query interface
- 📊 **Analytics and performance metrics**
- 🔗 **Knowledge graph generation**
- 🎨 **Document art generation**
- 💬 **Real-time collaboration via WebSocket**
- 🎵 **Voice query support** (placeholder for future)

### LLM Providers Supported
- **🚀 Groq** (Primary - Ultra-fast inference with Llama 3.1)
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Google (Gemini)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API Key (free tier available at https://console.groq.com/)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd knowledge-base-search-engine

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Configuration

Edit `.env` file and add your API keys:

```bash
# Required - Groq API Key (get free at https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# Optional - Other LLM providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# App Configuration
DEFAULT_LLM=groq  # Use Groq by default for best performance
```

### Running the Application

```bash
# Start the server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Open your browser
# - Main Interface: http://127.0.0.1:8000
# - API Documentation: http://127.0.0.1:8000/docs
# - Alternative UI: http://127.0.0.1:8000/crazy
```

## 📖 Usage

### 1. Upload Documents
- Navigate to the web interface
- Upload documents (PDF, DOCX, TXT, etc.)
- Wait for processing and indexing

### 2. Ask Questions
- Use the query interface to ask questions
- Get AI-powered answers with source citations
- Explore knowledge graphs and analytics

### 3. API Usage

```python
import requests

# Upload a document
with open('document.pdf', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})

# Ask a question
response = requests.post('http://localhost:8000/query', json={
    'question': 'What are the main topics discussed?',
    'max_results': 5,
    'include_sources': True
})

print(response.json()['answer'])
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   Groq API      │
│   (HTML/JS)     │◄──►│   Backend        │◄──►│   (LLM)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   ChromaDB       │
                       │   (Vector Store) │
                       └──────────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for LLM | Required |
| `DEFAULT_LLM` | Default LLM provider | `groq` |
| `GROQ_MODEL` | Groq model to use | `llama-3.1-8b-instant` |
| `CHUNK_SIZE` | Text chunk size for processing | `500` |
| `MAX_FILE_SIZE` | Maximum upload file size | `20MB` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |

## 📊 API Endpoints

### Core Endpoints
- `POST /upload` - Upload and process documents
- `POST /query` - Ask questions and get AI responses
- `GET /health` - System health check
- `GET /analytics` - Performance metrics

### Advanced Features
- `GET /knowledge-graph` - Generate knowledge graphs
- `POST /generate-doc-art/{document_id}` - Create document visualizations
- `WebSocket /ws/{user_id}` - Real-time collaboration

## 🧪 Testing

```bash
# Test Groq integration
python test_groq.py

# Run full test suite (if implemented)
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

MIT License - feel free to use this project for learning and development!

## 🆘 Support

- **Groq API Issues**: Check [Groq Documentation](https://console.groq.com/docs)
- **General Issues**: Open a GitHub issue
- **Performance**: Groq provides excellent speed - if you experience slowness, check your API limits

## 🎉 Why This Implementation is Awesome

1. **⚡ Lightning Fast**: Groq's inference speed is 10-20x faster than traditional APIs
2. **💰 Cost Effective**: Groq offers generous free tiers and competitive pricing
3. **🔧 Production Ready**: Comprehensive error handling, logging, and monitoring
4. **📈 Scalable**: Async processing, vector databases, and background tasks
5. **🎨 Feature Rich**: Goes beyond basic RAG with advanced visualizations and analytics

## 📋 Demo Video Content Ideas

When creating your demo video, showcase:

1. **Quick Setup** (2 minutes)
   - Show how to get a Groq API key
   - Demonstrate installation and configuration
   - Start the application

2. **Document Upload** (2 minutes)
   - Upload various document types
   - Show processing and indexing
   - Display document analytics

3. **Intelligent Querying** (3 minutes)
   - Ask complex questions across multiple documents
   - Show source citations and confidence scores
   - Demonstrate different query types

4. **Advanced Features** (3 minutes)
   - Show knowledge graph generation
   - Demonstrate real-time collaboration
   - Display analytics dashboard

5. **Performance Showcase** (1 minute)
   - Highlight Groq's fast response times
   - Compare with other providers if available
   - Show scalability metrics

---

**Built with ❤️ and powered by Groq's lightning-fast inference**