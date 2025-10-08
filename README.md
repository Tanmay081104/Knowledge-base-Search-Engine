# Knowledge-Based Search Engine

A sophisticated Retrieval-Augmented Generation (RAG) system that combines vector-based semantic search with multiple large language models to provide intelligent document-based question answering.

## Features

- **Multi-LLM Support**: Integrates with OpenAI GPT, Anthropic Claude, and Google Gemini
- **Vector Search**: Advanced semantic search using ChromaDB and sentence transformers
- **Document Processing**: Support for PDF, DOCX, TXT, MD, JSON, and image files with OCR
- **Real-time Analytics**: Comprehensive usage analytics and performance metrics
- **Knowledge Graph**: Visual representation of document relationships and entities
- **Document Art Generation**: AI-powered visual representations of document content
- **WebSocket Support**: Real-time collaboration and live updates
- **Professional Web Interface**: Clean, responsive frontend for document upload and querying

## Architecture

```
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   └── config.py        # Configuration management
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── services/
│       ├── rag_service.py           # RAG orchestration
│       ├── vector_store.py          # ChromaDB integration
│       ├── document_processor.py    # Document parsing
│       ├── knowledge_graph.py       # Graph generation
│       └── document_art_generator.py # Visual art creation
├── frontend/
│   └── templates/
│       └── index_clean.html # Web interface
└── documents/              # Sample documents
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

5. **Download SpaCy model**
```bash
python -m spacy download en_core_web_sm
```

## Configuration

Edit the `.env` file with your API keys:

```env
# Choose your preferred LLM provider
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Set default LLM (google, openai, or anthropic)
DEFAULT_LLM=google
```

## Usage

1. **Start the server**
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

2. **Access the web interface**
Open your browser and navigate to `http://127.0.0.1:8000`

3. **Upload documents**
Use the web interface to upload PDF, DOCX, TXT, or other supported document formats.

4. **Ask questions**
Enter questions about your uploaded documents and receive AI-powered answers with source citations.

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Key Endpoints

- `POST /upload` - Upload and process documents
- `POST /query` - Submit questions and receive AI-generated answers
- `GET /knowledge-graph` - Generate knowledge graph visualization
- `POST /generate-doc-art/{id}` - Create visual art from documents
- `GET /analytics` - Retrieve system analytics and metrics

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Vector Database**: ChromaDB with sentence-transformers
- **LLM Integration**: OpenAI API, Anthropic API, Google Gemini API
- **Document Processing**: PyPDF2, python-docx, pytesseract (OCR)
- **Knowledge Graph**: SpaCy, NetworkX
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Matplotlib, PIL for document art generation

## Performance Features

- Asynchronous processing for document uploads
- Background task queue for heavy operations
- Vector similarity search with cosine distance
- Intelligent chunking with overlap for context preservation
- Caching and connection pooling for optimal performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with FastAPI for high-performance API development
- ChromaDB for efficient vector similarity search
- Sentence Transformers for state-of-the-art embeddings
- Multiple LLM providers for diverse AI capabilities

## Support

For questions and support, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).
