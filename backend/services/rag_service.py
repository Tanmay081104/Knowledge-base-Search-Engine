"""RAG (Retrieval-Augmented Generation) service combining document retrieval with LLM generation."""

import time
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

import openai
import anthropic
import google.generativeai as genai
from loguru import logger

from backend.app.config import settings
from backend.services.vector_store import VectorStore
class RAGService:
    """Service for handling RAG queries using retrieved documents and LLM generation."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.default_llm = settings.default_llm
        
        # Initialize LLM clients
        self._init_llm_clients()
    
    def _init_llm_clients(self):
        """Initialize LLM clients based on configuration."""
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        
        try:
            if settings.openai_api_key and settings.openai_api_key != "demo_key":
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.info("OpenAI client not initialized (demo key)")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
        
        try:
            if settings.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
        
        try:
            if settings.google_api_key:
                genai.configure(api_key=settings.google_api_key)
                self.google_client = genai.GenerativeModel(settings.google_model)
                logger.info(f"Google Gemini client initialized with model: {settings.google_model}")
            else:
                logger.info("Google Gemini client not initialized (no API key)")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Gemini client: {str(e)}")
    
    async def process_query(
        self,
        question: str,
        max_results: int = 5,
        include_sources: bool = True,
        llm_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query using RAG approach.
        
        Args:
            question: User question
            max_results: Maximum number of retrieved documents
            include_sources: Whether to include source information
            llm_provider: LLM provider to use (openai, anthropic, or None for default)
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing query: {question[:100]}...")
            similar_chunks = await self.vector_store.search_similar_chunks(
                query=question,
                max_results=max_results
            )
            
            if not similar_chunks:
                return {
                    'success': False,
                    'question': question,
                    'answer': "I couldn't find any relevant documents to answer your question. Please try uploading some documents first or rephrasing your question.",
                    'sources': [],
                    'processing_time': time.time() - start_time,
                    'timestamp': datetime.now()
                }
            
            # Step 2: Generate answer using LLM
            llm_to_use = llm_provider or self.default_llm
            answer = await self._generate_answer(question, similar_chunks, llm_to_use)
            
            # Step 3: Prepare response
            sources = []
            if include_sources:
                sources = self._format_sources(similar_chunks)
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'question': question,
                'answer': answer,
                'sources': sources,
                'processing_time': processing_time,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'success': False,
                'question': question,
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now()
            }
    
    async def _generate_answer(
        self,
        question: str,
        similar_chunks: List[Dict[str, Any]],
        llm_provider: str
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            question: User question
            similar_chunks: Retrieved document chunks
            llm_provider: LLM provider to use
            
        Returns:
            Generated answer
        """
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(similar_chunks, 1):
            context_parts.append(f"Document {i} (from {chunk['filename']}):")
            context_parts.append(chunk['content'])
            context_parts.append("")  # Empty line for separation
        
        context = "\n".join(context_parts)
        
        # Create the prompt
        prompt = self._create_rag_prompt(question, context)
        
        # Generate answer using specified LLM with fallback to demo mode
        try:
            if llm_provider == "openai" and self.openai_client:
                return await self._generate_openai_answer(prompt)
            elif llm_provider == "anthropic" and self.anthropic_client:
                return await self._generate_anthropic_answer(prompt)
            elif llm_provider == "google" and self.google_client:
                return await self._generate_google_answer(prompt)
            else:
                # No LLM available - use demo mode
                return await self._generate_demo_answer(question, similar_chunks)
        except Exception as e:
            logger.warning(f"LLM generation failed, falling back to demo mode: {str(e)[:100]}")
            # Fallback to demo mode when LLM fails
            return await self._generate_demo_answer(question, similar_chunks)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a well-structured prompt for RAG."""
        return f"""Using these documents, answer the user's question succinctly.

Context Documents:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the provided documents
- If the documents don't contain enough information to answer the question completely, acknowledge this
- Cite specific information from the documents when relevant
- Keep the answer concise but comprehensive
- If multiple documents provide relevant information, synthesize them appropriately

Answer:"""
    
    async def _generate_openai_answer(self, prompt: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating OpenAI answer: {str(e)}")
            raise
    
    async def _generate_anthropic_answer(self, prompt: str) -> str:
        """Generate answer using Anthropic API."""
        try:
            response = self.anthropic_client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating Anthropic answer: {str(e)}")
            raise
    
    async def _generate_google_answer(self, prompt: str) -> str:
        """Generate answer using Google Gemini API."""
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_tokens,
            }
            
            # Generate response
            response = await asyncio.to_thread(
                self.google_client.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("Google Gemini returned empty response")
            
        except Exception as e:
            logger.error(f"Error generating Google Gemini answer: {str(e)}")
            raise
    
    def _format_sources(self, similar_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for response."""
        sources = []
        
        for chunk in similar_chunks:
            source = {
                'document_id': chunk['document_id'],
                'filename': chunk['filename'],
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                'similarity_score': chunk['similarity_score']
            }
            sources.append(source)
        
        return sources
    
    async def get_available_llms(self) -> List[Dict[str, Any]]:
        """Get list of available LLM providers."""
        providers = []
        
        if self.openai_client:
            providers.append({
                'provider': 'openai',
                'model': settings.openai_model,
                'status': 'available'
            })
        
        if self.anthropic_client:
            providers.append({
                'provider': 'anthropic',
                'model': settings.anthropic_model,
                'status': 'available'
            })
        
        if self.google_client:
            providers.append({
                'provider': 'google',
                'model': settings.google_model,
                'status': 'available'
            })
        
        return providers
    
    async def test_llm_connection(self, provider: str) -> Dict[str, Any]:
        """Test connection to an LLM provider."""
        try:
            test_prompt = "This is a test. Please respond with 'Test successful'."
            
            if provider == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": test_prompt}],
                    max_tokens=50,
                    temperature=0
                )
                return {
                    'provider': provider,
                    'status': 'success',
                    'response': response.choices[0].message.content.strip()
                }
                
            elif provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=50,
                    temperature=0,
                    messages=[{"role": "user", "content": test_prompt}]
                )
                return {
                    'provider': provider,
                    'status': 'success',
                    'response': response.content[0].text.strip()
                }
            else:
                return {
                    'provider': provider,
                    'status': 'error',
                    'error': f'Provider {provider} not configured or not available'
                }
                
        except Exception as e:
            return {
                'provider': provider,
                'status': 'error',
                'error': str(e)
            }
    
    async def _generate_demo_answer(self, question: str, similar_chunks: List[Dict[str, Any]]) -> str:
        """Generate demo answer when LLM is not available."""
        if not similar_chunks:
            return "I couldn't find any relevant information in the knowledge base to answer your question. This is running in DEMO mode - please add your API keys to .env file for full AI-powered responses!"
        
        # Create a simple response based on the most relevant chunk
        best_chunk = similar_chunks[0]
        content_preview = best_chunk['content'][:300] + "..." if len(best_chunk['content']) > 300 else best_chunk['content']
        
        demo_response = f"🤖 **DEMO MODE RESPONSE** 🤖\n\n"
        demo_response += f"Based on the document '{best_chunk['filename']}', here's what I found:\n\n"
        demo_response += f"{content_preview}\n\n"
        demo_response += f"📊 **Search Results**: Found {len(similar_chunks)} relevant chunks\n"
        demo_response += f"🎯 **Similarity Score**: {best_chunk['similarity_score']:.2%}\n\n"
        demo_response += f"💡 **Note**: This is a demo response! Add your OpenAI or Anthropic API key to .env file for full AI-powered answers with advanced reasoning and synthesis."
        
        return demo_response
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_collection_stats()
    
    async def add_document_to_knowledge_base(self, document_data: Dict[str, Any]) -> int:
        """Add a document to the knowledge base."""
        return await self.vector_store.add_document(document_data)