"""
Simplified Knowledge Base Search Engine with Groq Integration
This version bypasses complex dependencies and focuses on core RAG functionality.
"""

import os
import tempfile
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import PyPDF2
from groq import Groq

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = ["pdf", "txt", "md"]

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Base Search Engine",
    description="RAG system with Groq integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Simple in-memory storage for documents
documents_store: Dict[str, Dict] = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    max_results: int = 3

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    processing_time: float
    timestamp: datetime

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    filename: str
    chunks_created: int

# HTML Frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Base Search Engine</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #333; 
            text-align: center;
            margin-bottom: 30px;
        }
        .section { 
            margin-bottom: 30px; 
            padding: 20px; 
            border: 1px solid #ddd; 
            border-radius: 8px;
            background-color: #fafafa;
        }
        .upload-area { 
            border: 2px dashed #007bff; 
            padding: 20px; 
            text-align: center; 
            border-radius: 8px;
            background-color: white;
        }
        input[type="file"] { 
            margin: 10px 0; 
        }
        button { 
            background-color: #007bff; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 5px;
        }
        button:hover { 
            background-color: #0056b3; 
        }
        textarea { 
            width: 100%; 
            height: 60px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 10px;
            resize: vertical;
        }
        .answer { 
            background-color: #e8f5e8; 
            padding: 15px; 
            border-radius: 5px; 
            margin-top: 10px;
            border-left: 4px solid #28a745;
        }
        .error { 
            background-color: #f8d7da; 
            color: #721c24; 
            padding: 15px; 
            border-radius: 5px; 
            margin-top: 10px;
            border-left: 4px solid #dc3545;
        }
        .success { 
            background-color: #d4edda; 
            color: #155724; 
            padding: 15px; 
            border-radius: 5px; 
            margin-top: 10px;
            border-left: 4px solid #28a745;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Knowledge Base Search Engine</h1>
        <p style="text-align: center; color: #666;"></p>
        
        <div class="section">
            <h2>📄 Upload Documents</h2>
            <div class="upload-area">
                <p>Upload your documents (PDF, TXT, MD files)</p>
                <input type="file" id="fileInput" accept=".pdf,.txt,.md" />
                <br>
                <button onclick="uploadDocument()">Upload Document</button>
            </div>
            <div id="uploadResult"></div>
        </div>
        
        <div class="section">
            <h2>❓ Ask Questions</h2>
            <textarea id="questionInput" placeholder="Ask a question about your uploaded documents..."></textarea>
            <br>
            <button onclick="askQuestion()">Get Answer</button>
            <div id="queryResult"></div>
        </div>
        
        <div class="section">
            <h2>📊 System Status</h2>
            <button onclick="checkHealth()">Check System Health</button>
            <div id="healthResult"></div>
        </div>
    </div>

    <script>
        async function uploadDocument() {
            const fileInput = document.getElementById('fileInput');
            const resultDiv = document.getElementById('uploadResult');
            
            if (!fileInput.files[0]) {
                resultDiv.innerHTML = '<div class="error">Please select a file to upload.</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            resultDiv.innerHTML = '<div class="loading"></div> Uploading and processing document...';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = `<div class="success">
                        ✅ ${result.message}<br>
                        Document ID: ${result.document_id}<br>
                        Chunks created: ${result.chunks_created}
                    </div>`;
                } else {
                    resultDiv.innerHTML = `<div class="error">❌ Upload failed: ${result.message || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const resultDiv = document.getElementById('queryResult');
            
            if (!questionInput.value.trim()) {
                resultDiv.innerHTML = '<div class="error">Please enter a question.</div>';
                return;
            }
            
            resultDiv.innerHTML = '<div class="loading"></div> Processing your question with Groq...';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        question: questionInput.value,
                        max_results: 3
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    let sourcesHtml = '';
                    if (result.sources && result.sources.length > 0) {
                        sourcesHtml = '<h4>📚 Sources:</h4><ul>';
                        result.sources.forEach(source => {
                            sourcesHtml += `<li><strong>${source.filename}</strong>: ${source.content.substring(0, 200)}...</li>`;
                        });
                        sourcesHtml += '</ul>';
                    }
                    
                    resultDiv.innerHTML = `<div class="answer">
                        <h4>🤖 Answer:</h4>
                        <p>${result.answer}</p>
                        ${sourcesHtml}
                        <small>Processing time: ${result.processing_time.toFixed(2)}s</small>
                    </div>`;
                } else {
                    resultDiv.innerHTML = `<div class="error">❌ Query failed: ${result.answer || 'Unknown error'}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function checkHealth() {
            const resultDiv = document.getElementById('healthResult');
            
            try {
                const response = await fetch('/health');
                const result = await response.json();
                
                resultDiv.innerHTML = `<div class="stats">
                    <strong>Status:</strong> ${result.status}<br>
                    <strong>Version:</strong> ${result.version}<br>
                    <strong>Documents:</strong> ${Object.keys(window.documentsStore || {}).length} loaded<br>
                    <strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}
                </div>`;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Health check failed: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

# Utility functions
def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from uploaded file"""
    extension = Path(filename).suffix.lower().lstrip('.')
    
    if extension == 'pdf':
        return extract_pdf_text(file_path)
    elif extension in ['txt', 'md']:
        return extract_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF file"""
    text_content = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
        return text_content.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_file(file_path: str) -> str:
    """Extract text from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            raise ValueError(f"Failed to read text file: {str(e)}")

def create_chunks(text: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
    """Create text chunks"""
    if not text.strip():
        return []
    
    # Simple chunking by sentences
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'content': current_chunk.strip(),
                'length': len(current_chunk)
            })
            current_chunk = sentence
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append({
            'chunk_id': str(uuid.uuid4()),
            'content': current_chunk.strip(),
            'length': len(current_chunk)
        })
    
    return chunks

def simple_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Simple keyword-based search through documents"""
    results = []
    query_words = query.lower().split()
    
    for doc_id, doc_data in documents_store.items():
        for chunk in doc_data['chunks']:
            content_lower = chunk['content'].lower()
            score = sum(1 for word in query_words if word in content_lower)
            
            if score > 0:
                results.append({
                    'document_id': doc_id,
                    'filename': doc_data['filename'],
                    'chunk_id': chunk['chunk_id'],
                    'content': chunk['content'],
                    'score': score / len(query_words)  # Normalize score
                })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_results]

async def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate answer using Groq"""
    if not context_chunks:
        return "I couldn't find any relevant information in the uploaded documents to answer your question. Please upload some documents first."
    
    # Prepare context
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(f"Document {i} (from {chunk['filename']}):")
        context_parts.append(chunk['content'])
        context_parts.append("")
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Using these documents, answer the user's question succinctly.

Context Documents:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the provided documents
- If the documents don't contain enough information, acknowledge this
- Keep the answer concise but comprehensive
- Cite specific information from the documents when relevant

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer with Groq: {str(e)}"

# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend"""
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": f"🚀 Healthy! {len(documents_store)} documents loaded",
        "timestamp": datetime.now(),
        "version": "1.0.0-Simplified",
        "groq_model": GROQ_MODEL
    }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    # Validate file
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    extension = Path(file.filename).suffix.lower().lstrip('.')
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract text
        text_content = extract_text_from_file(tmp_file_path, file.filename)
        
        # Create chunks
        chunks = create_chunks(text_content)
        
        # Store document
        document_id = str(uuid.uuid4())
        documents_store[document_id] = {
            'document_id': document_id,
            'filename': file.filename,
            'text_content': text_content,
            'chunks': chunks,
            'upload_time': datetime.now().isoformat()
        }
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return DocumentUploadResponse(
            success=True,
            message=f"Document {file.filename} processed successfully!",
            document_id=document_id,
            filename=file.filename,
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        # Clean up temp file if it exists
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query"""
    import time
    start_time = time.time()
    
    try:
        # Search for relevant chunks
        relevant_chunks = simple_search(request.question, request.max_results)
        
        # Generate answer
        answer = await generate_answer(request.question, relevant_chunks)
        
        # Prepare sources
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                'document_id': chunk['document_id'],
                'filename': chunk['filename'],
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                'score': chunk['score']
            })
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            success=True,
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            question=request.question,
            answer=f"Error processing query: {str(e)}",
            sources=[],
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )

@app.get("/api")
async def api_info():
    """API information"""
    return {
        "message": "🚀 Knowledge Base Search Engine API with Groq Integration",
        "status": "Active and Ready!",
        "features": [
            "📄 Document Upload (PDF, TXT, MD)",
            "🤖 Groq-Powered Question Answering",
            "🔍 Intelligent Document Search",
            "⚡ Lightning-Fast AI Responses"
        ],
        "documents_loaded": len(documents_store),
        "groq_model": GROQ_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Knowledge Base Search Engine with Groq Integration!")
    print("🔥 Simplified version - Fast and reliable!")
    
    uvicorn.run(
        "simple_main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )