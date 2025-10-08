"""Vector database service using ChromaDB for document embeddings."""

import os
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Using mock embeddings for demo.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
    class MockSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name
            
        def encode(self, texts):
            """Create mock embeddings for demo purposes."""
            import random
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            # Create consistent mock embeddings (384 dimensions for MiniLM)
            embeddings = []
            for text in texts:
                # Use hash of text for consistent embeddings
                seed = hash(text) % 1000000
                random.seed(seed)
                embedding = [random.uniform(-1, 1) for _ in range(384)]
                embeddings.append(embedding)
            
            # Return as numpy-like array to match SentenceTransformer behavior
            class MockArray:
                def __init__(self, data):
                    self.data = data
                def tolist(self):
                    return self.data if isinstance(self.data[0], list) else [self.data]
            
            return MockArray(embeddings if len(texts) > 1 else embeddings[0])
    
    SentenceTransformer = MockSentenceTransformer

from backend.app.config import settings

class VectorStore:
    """Service for managing document embeddings using ChromaDB."""
    
    def __init__(self):
        self.embedding_model_name = settings.embedding_model
        self.persist_directory = settings.chroma_persist_directory
        self.collection_name = "knowledge_base"
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB client
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    async def add_document(self, document_data: Dict[str, Any]) -> int:
        """
        Add a document and its chunks to the vector store.
        
        Args:
            document_data: Document information with chunks
            
        Returns:
            Number of chunks added
        """
        try:
            document_id = document_data['document_id']
            filename = document_data['filename']
            chunks = document_data['chunks']
            upload_time = datetime.now().isoformat()
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = chunk['chunk_id']
                content = chunk['content']
                
                chunk_ids.append(chunk_id)
                chunk_texts.append(content)
                metadatas.append({
                    'document_id': document_id,
                    'filename': filename,
                    'chunk_length': chunk['length'],
                    'upload_time': upload_time
                })
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
            embeddings = self.embedding_model.encode(chunk_texts).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Added document {filename} with {len(chunks)} chunks to vector store")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    async def search_similar_chunks(
        self, 
        query: str, 
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks based on query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            similar_chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunk_data = {
                        'chunk_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'document_id': results['metadatas'][0][i]['document_id'],
                        'filename': results['metadatas'][0][i]['filename']
                    }
                    similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks for query")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get unique documents
            if count > 0:
                results = self.collection.get(include=['metadatas'])
                unique_docs = set()
                for metadata in results['metadatas']:
                    unique_docs.add(metadata['document_id'])
                unique_document_count = len(unique_docs)
            else:
                unique_document_count = 0
            
            # Calculate approximate database size
            db_size = self._calculate_db_size()
            
            stats = {
                'total_chunks': count,
                'total_documents': unique_document_count,
                'embedding_model': self.embedding_model_name,
                'database_size': db_size,
                'collection_name': self.collection_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'total_chunks': 0,
                'total_documents': 0,
                'embedding_model': self.embedding_model_name,
                'database_size': '0 MB',
                'collection_name': self.collection_name
            }
    
    def _calculate_db_size(self) -> str:
        """Calculate approximate database size."""
        try:
            total_size = 0
            for root, dirs, files in os.walk(self.persist_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            
            # Convert to human readable format
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            else:
                return f"{total_size / (1024 * 1024):.1f} MB"
                
        except Exception:
            return "Unknown"