"""Knowledge Base Search Engine - Advanced RAG System"""

import os
import tempfile
import asyncio
from typing import List, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.websockets import WebSocket
from fastapi.templating import Jinja2Templates
import json

from backend.app.config import settings
from backend.models.schemas import *
from backend.services.document_processor import DocumentProcessor
from backend.services.rag_service import RAGService
from backend.services.knowledge_graph import KnowledgeGraphService
from backend.services.document_art_generator import DocumentArtGenerator
from loguru import logger

app = FastAPI(
    title="Knowledge Base Search Engine",
    description="Advanced RAG system with multi-LLM support and semantic search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor = DocumentProcessor()
rag_service = RAGService()
knowledge_graph_service = KnowledgeGraphService()
document_art_generator = DocumentArtGenerator()
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_sessions = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_sessions[user_id] = websocket
        await self.broadcast(f"🎉 User {user_id} joined the knowledge party!")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections.remove(websocket)
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the clean professional frontend interface"""
    try:
        with open("frontend/templates/index_clean.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Knowledge Base Search Engine</h1><p>Error loading frontend: {str(e)}</p>",
            status_code=500
        )

@app.get("/crazy", response_class=HTMLResponse)
async def serve_crazy_frontend():
    """Serve the original crazy frontend with all the graphics"""
    try:
        with open("frontend/templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Advanced Frontend Error</h1><p>Error loading crazy frontend: {str(e)}</p>",
            status_code=500
        )

@app.get("/test", response_class=HTMLResponse)
async def serve_test_frontend():
    """Professional implementation"""
    try:
        with open("test_frontend.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Test Page Error</h1><p>Error loading test page: {str(e)}</p>",
            status_code=500
        )

@app.get("/api", response_model=dict)
async def api_info():
    """Professional implementation"""
    return {
        "message": "🚀 Welcome to the Advanced Knowledge Base Search Engine API! 🤖",
        "status": "ABSOLUTELY Professional! 🔥",
        "features": [
            "🤖 Multi-LLM RAG System",
            "🎯 Vector Semantic Search", 
            "🎮 Gamified Learning",
            "🔊 Voice Queries",
            "🌈 Sentiment Analysis",
            "🔗 Knowledge Graphs",
            "⚡ Real-time Collaboration",
            "🎨 AI Document Art"
        ],
        "easter_egg": "Try asking me something Advanced! 🎊"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Professional implementation"""
    stats = rag_service.get_vector_store_stats()
    return HealthResponse(
        status=f"🚀 Advanced HEALTHY! {stats['total_documents']} docs loaded! 🔥",
        timestamp=datetime.now(),
        version="2.0.0-Advanced"
    )

# 📁 Professional DOCUMENT UPLOAD WITH PROCESSING Processing
@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Professional implementation"""
    
    # Validate file like a boss
    if not document_processor.validate_file(file.filename, file.size or 0):
        raise HTTPException(
            status_code=400, 
            detail=f"🚫 File validation failed! Check size and format for {file.filename}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
                document_data = await document_processor.process_uploaded_file(
            tmp_file_path, 
            file.filename
        )
        
                background_tasks.add_task(
            add_to_knowledge_base_background,
            document_data,
            tmp_file_path
        )
        
                await manager.broadcast(f"🎉 New document uploaded: {file.filename}! Knowledge level INCREASED! 📈")
        
        return DocumentUploadResponse(
            success=True,
            message=f"🚀 Document {file.filename} processed with Advanced efficiency!",
            document_id=document_data['document_id'],
            filename=file.filename,
            chunks_created=document_data['chunks_count']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"🔥 Upload failed with Advanced error: {str(e)}"
        )

async def add_to_knowledge_base_background(document_data: dict, tmp_file_path: str):
    """Background task to add document to knowledge base"""
    try:
        await rag_service.add_document_to_knowledge_base(document_data)
        os.unlink(tmp_file_path)  # Clean up temp file
        await manager.broadcast(f"✅ Document {document_data['filename']} fully indexed! Ready for Advanced queries! 🤖")
    except Exception as e:
        await manager.broadcast(f"❌ Error indexing {document_data['filename']}: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Professional implementation"""
    
    # Add some Advanced pre-processing
    enhanced_question = await enhance_query_with_ai_magic(request.question)
    
    # Process with our Professional RAG system
    result = await rag_service.process_query(
        question=enhanced_question,
        max_results=request.max_results,
        include_sources=request.include_sources
    )
    
    # Add some Advanced post-processing
    if result['success']:
        result['answer'] = await add_emoji_magic(result['answer'])
        result['mood'] = await analyze_answer_mood(result['answer'])
        result['crazy_factor'] = calculate_crazy_factor(result['answer'])
    
    # Broadcast query activity! 📢
    await manager.broadcast(f"🔍 Someone asked: '{request.question[:50]}...' - AI is thinking Advanced thoughts! 🤖")
    
    return QueryResponse(**result)

async def enhance_query_with_ai_magic(question: str) -> str:
    """Professional implementation"""
    # Add context hints for better retrieval
    if "?" not in question:
        question += "?"
    
    # Add some Advanced enhancement logic here
    enhanced = f"[ENHANCED QUERY] {question}"
    return enhanced

async def add_emoji_magic(answer: str) -> str:
    """Add Advanced emoji magic to answers! 🎨"""
    import re
    
    # Add emojis based on content
    if re.search(r'\b(good|great|excellent|amazing)\b', answer.lower()):
        answer = f"✅ {answer}"
    if re.search(r'\b(problem|error|issue|difficult)\b', answer.lower()):
        answer = f"⚠️ {answer}"
    if re.search(r'\b(learn|study|education)\b', answer.lower()):
        answer = f"📚 {answer}"
    if re.search(r'\b(technology|AI|artificial intelligence)\b', answer.lower()):
        answer = f"🤖 {answer}"
    
    return answer

async def analyze_answer_mood(answer: str) -> str:
    """Professional implementation"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'difficult', 'problem']
    
    answer_lower = answer.lower()
    positive_count = sum(1 for word in positive_words if word in answer_lower)
    negative_count = sum(1 for word in negative_words if word in answer_lower)
    
    if positive_count > negative_count:
        return "😊 Positive & Optimistic"
    elif negative_count > positive_count:
        return "😔 Concerned & Thoughtful"
    else:
        return "😐 Neutral & Balanced"

def calculate_crazy_factor(answer: str) -> float:
    """Professional implementation"""
    crazy_words = ['amazing', 'incredible', 'fantastic', 'revolutionary', 'breakthrough', 'innovative']
    crazy_count = sum(1 for word in crazy_words if word.lower() in answer.lower())
    return min(crazy_count / len(answer.split()) * 100, 100.0)

# 🔊 VOICE QUERY SUPPORT (Placeholder for Advanced voice features)
@app.post("/voice-query")
async def voice_query(audio_file: UploadFile = File(...)):
    """Professional implementation"""
    return {
        "message": "🎤 Voice processing coming soon! The AI will HEAR your thoughts! 🔊",
        "status": "CRAZY_DEVELOPMENT_MODE",
        "filename": audio_file.filename
    }

@app.get("/achievements")
async def get_achievements():
    """Get Advanced achievements and badges! 🏆"""
    return {
        "achievements": [
            {"name": "🚀 First Query", "description": "Asked your first Advanced question!", "unlocked": True},
            {"name": "📚 Document Master", "description": "Uploaded 10+ documents!", "unlocked": False},
            {"name": "🤖 AI Whisperer", "description": "Had 100+ conversations with AI!", "unlocked": False},
            {"name": "🔥 Crazy Explorer", "description": "Discovered hidden knowledge gems!", "unlocked": False}
        ],
        "total_points": 1337,
        "level": "Advanced BEGINNER 🎊",
        "next_level": "Professional EXPLORER 🚀"
    }

@app.get("/knowledge-graph")
async def get_knowledge_graph():
    try:
        # Get all documents from vector store
        vector_stats = rag_service.get_vector_store_stats()
        
        if vector_stats['total_documents'] == 0:
            return {
                "message": "🤖 No documents found! Upload some docs to see the Advanced knowledge connections! 📚",
                "graph": {"nodes": [], "edges": []},
                "summary": {"total_nodes": 0, "total_edges": 0}
            }
        
        # Get actual documents from the vector store for analysis
        documents = await get_documents_for_graph_analysis()
        
        if not documents:
            return {
                "message": "📄 Documents detected but content not accessible for graph analysis! 🔍",
                "mock_data": True,
                "graph": await create_mock_knowledge_graph(vector_stats)
            }
        
        # Generate real knowledge graph
        knowledge_graph = await knowledge_graph_service.analyze_documents(documents)
        
        return {
            "message": "🚀 Advanced knowledge graph generated with AI-powered analysis! 🤖✨",
            "real_data": True,
            **knowledge_graph
        }
        
    except Exception as e:
        logger.error(f"Error generating knowledge graph: {str(e)}")
        # Fallback to enhanced mock data
        stats = rag_service.get_vector_store_stats()
        return {
            "message": f"⚡ Generating enhanced knowledge graph... {str(e)[:50]}",
            "fallback": True,
            "graph": await create_mock_knowledge_graph(stats)
        }

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Professional implementation"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "query":
                # Process query and broadcast result
                result = await rag_service.process_query(
                    question=message_data["question"],
                    max_results=3,
                    include_sources=True
                )
                await manager.broadcast(f"🔍 {user_id} asked: {message_data['question']}")
                await manager.send_personal_message(
                    json.dumps({"type": "answer", "result": result}),
                    websocket
                )
            
            elif message_data["type"] == "chat":
                await manager.broadcast(f"💬 {user_id}: {message_data['message']}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        await manager.broadcast(f"👋 {user_id} left the knowledge party!")

@app.get("/analytics")
async def get_analytics():
    """Professional implementation"""
    stats = rag_service.get_vector_store_stats()
    
    return {
        "overview": {
            "total_documents": stats['total_documents'],
            "total_chunks": stats['total_chunks'],
            "database_size": stats['database_size'],
            "embedding_model": stats['embedding_model'],
            "status": "ABSOLUTELY Professional! 🚀"
        },
        "performance": {
            "queries_per_minute": 42,  # Mock data for now
            "average_response_time": "0.1337 seconds ⚡",
            "accuracy_score": "99.9% Advanced ACCURATE! 🎯",
            "user_satisfaction": "THROUGH THE ROOF! 📈"
        },
        "trends": {
            "most_asked_topic": "🤖 Artificial Intelligence",
            "peak_hours": "ALWAYS PEAK TIME! 🔥",
            "growth_rate": "+1337% per day! 📈"
        },
        "crazy_metrics": {
            "mind_blown_count": 9001,
            "eureka_moments": 1337,
            "knowledge_explosions": 42,
            "ai_happiness_level": "MAXIMUM OVERDRIVE! 🤖✨"
        }
    }

# 🎨 AI DOCUMENT ART GENERATOR (Advanced FEATURE!)
@app.post("/generate-doc-art/{document_id}")
async def generate_document_art(
    document_id: str,
    style: str = "auto",
    color_palette: str = "vibrant",
    size: str = "large"
):
    """Professional implementation"""
    try:
        # Get document data from vector store
        document_data = await get_document_for_art_generation(document_id)
        
        if not document_data:
            return {
                "success": False,
                "message": f"📄 Document {document_id} not found! Upload it first to create ART! 🎨",
                "error": "Document not found"
            }
        
        # Generate the Advanced art!
        art_result = await document_art_generator.generate_document_art(
            document_data=document_data,
            style=style,
            color_palette=color_palette,
            size=size
        )
        
        return art_result
        
    except Exception as e:
        logger.error(f"Error generating document art: {str(e)}")
        return {
            "success": False,
            "message": f"🎨 Art generation failed: {str(e)}",
            "error": str(e)
        }

@app.get("/art-styles")
async def get_art_styles():
    """Professional implementation"""
    try:
        styles = await document_art_generator.get_available_styles()
        palettes = await document_art_generator.get_available_palettes()
        
        return {
            "success": True,
            "message": "🎨 Advanced art styles ready for your documents! ✨",
            "styles": styles,
            "color_palettes": palettes,
            "sizes": [
                {"size": "small", "name": "📏 Small (800x600)", "description": "Quick preview size"},
                {"size": "medium", "name": "🖼️ Medium (1200x900)", "description": "Standard quality"},
                {"size": "large", "name": "🗺️ Large (1600x1200)", "description": "High-resolution masterpiece"}
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting art styles: {str(e)}",
            "error": str(e)
        }

# 🔧 HELPER FUNCTIONS FOR KNOWLEDGE GRAPH
async def get_documents_for_graph_analysis():
    """Get documents from vector store for knowledge graph analysis."""
    try:
        # This is a simplified approach - in a real system you'd want to 
        # retrieve actual document content from the vector store
        vector_store = rag_service.vector_store
        if vector_store.collection.count() == 0:
            return []
        
        # Get all documents with their metadata and content
        results = vector_store.collection.get(
            include=['documents', 'metadatas']
        )
        
        if not results or not results['documents']:
            return []
        
        # Group chunks by document
        documents = {}
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            doc_id = metadata['document_id']
            filename = metadata['filename']
            
            if doc_id not in documents:
                documents[doc_id] = {
                    'document_id': doc_id,
                    'filename': filename,
                    'content': [],
                    'upload_time': metadata.get('upload_time', ''),
                }
            
            documents[doc_id]['content'].append(doc)
        
        # Combine chunks into full document content
        processed_documents = []
        for doc_id, doc_data in documents.items():
            processed_documents.append({
                'document_id': doc_id,
                'filename': doc_data['filename'],
                'content': ' '.join(doc_data['content'])[:50000],  # Limit content length
                'upload_time': doc_data['upload_time']
            })
        
        logger.info(f"Retrieved {len(processed_documents)} documents for graph analysis")
        return processed_documents
        
    except Exception as e:
        logger.error(f"Error getting documents for graph analysis: {str(e)}")
        return []

async def create_mock_knowledge_graph(stats):
    """Create an enhanced mock knowledge graph when real analysis isn't available."""Professional implementation"""Get document data for art generation."""
    try:
        # Get document chunks from vector store
        vector_store = rag_service.vector_store
        
        # Try to find chunks with this document_id
        results = vector_store.collection.get(
            where={"document_id": document_id},
            include=['documents', 'metadatas']
        )
        
        if not results or not results['documents']:
            logger.warning(f"No chunks found for document_id: {document_id}")
            return None
        
        # Combine all chunks into full document
        all_chunks = results['documents']
        metadata = results['metadatas'][0] if results['metadatas'] else {}
        
        # Create document data for art generation
        document_data = {
            'document_id': document_id,
            'filename': metadata.get('filename', f'document_{document_id}'),
            'content': ' '.join(all_chunks)[:20000],  # Limit content length for performance
            'upload_time': metadata.get('upload_time', ''),
            'chunk_count': len(all_chunks)
        }
        
        logger.info(f"Retrieved document {document_data['filename']} for art generation ({len(all_chunks)} chunks)")
        return document_data
        
    except Exception as e:
        logger.error(f"Error getting document for art generation: {str(e)}")
        return None

if __name__ == "__main__":
    print("🚀 LAUNCHING THE CRAZIEST KNOWLEDGE BASE EVER CREATED! 🚀")
    print("🔥 Prepare for MAXIMUM KNOWLEDGE OVERDRIVE! 🔥")
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info"
    )