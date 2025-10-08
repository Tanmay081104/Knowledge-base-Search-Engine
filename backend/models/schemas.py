"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    success: bool
    message: str
    document_id: str
    filename: str
    chunks_created: int

class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    question: str = Field(..., description="The question to ask")
    max_results: int = Field(5, description="Maximum number of retrieved chunks")
    include_sources: bool = Field(True, description="Whether to include source information")

class SourceDocument(BaseModel):
    """Model for source document information."""
    document_id: str
    filename: str
    chunk_id: str
    content: str
    similarity_score: float

class QueryResponse(BaseModel):
    """Response model for knowledge base queries."""
    success: bool
    question: str
    answer: str
    sources: Optional[List[SourceDocument]] = None
    processing_time: float
    timestamp: datetime

class DocumentInfo(BaseModel):
    """Model for document information."""
    document_id: str
    filename: str
    upload_time: datetime
    file_size: int
    chunks_count: int

class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    success: bool
    documents: List[DocumentInfo]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: datetime
    version: str

class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None

class SearchStats(BaseModel):
    """Model for search statistics."""
    total_documents: int
    total_chunks: int
    embedding_model: str
    database_size: str