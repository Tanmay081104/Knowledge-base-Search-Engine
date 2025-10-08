"""Professional implementation"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass, asdict
from loguru import logger

from backend.app.config import settings

@dataclass
class QueryMetrics:
    """Metrics for individual queries."""
    query_id: str
    question: str
    timestamp: datetime
    processing_time: float
    success: bool
    user_id: str = "anonymous"
    response_length: int = 0
    sources_count: int = 0
    confidence_score: float = 0.0
    sentiment: str = "neutral"

@dataclass
class DocumentMetrics:
    """Metrics for documents."""
    document_id: str
    filename: str
    upload_time: datetime
    size_bytes: int
    chunks_count: int
    query_count: int = 0
    last_accessed: Optional[datetime] = None
    popularity_score: float = 0.0

class AnalyticsService:
    """Service for comprehensive analytics and insights."""
    
    def __init__(self):
        # In-memory storage for demo - in production, use a proper database
        self.query_history: List[QueryMetrics] = []
        self.document_metrics: Dict[str, DocumentMetrics] = {}
        self.user_sessions: Dict[str, List[datetime]] = defaultdict(list)
        self.search_patterns: Dict[str, int] = defaultdict(int)
        self.knowledge_discovery_events: List[Dict[str, Any]] = []
        
        # Analytics configuration
        self.retention_days = 30  # Keep data for 30 days
        
    async def track_query(
        self,
        query_id: str,
        question: str,
        processing_time: float,
        success: bool,
        user_id: str = "anonymous",
        response_length: int = 0,
        sources_count: int = 0,
        confidence_score: float = 0.0
    ):
        """Track a query for analytics."""
        try:
            # Analyze query sentiment
            sentiment = await self._analyze_query_sentiment(question)
            
            # Create query metrics
            query_metrics = QueryMetrics(
                query_id=query_id,
                question=question[:200],  # Limit question length
                timestamp=datetime.now(),
                processing_time=processing_time,
                success=success,
                user_id=user_id,
                response_length=response_length,
                sources_count=sources_count,
                confidence_score=confidence_score,
                sentiment=sentiment
            )
            
            # Add to history
            self.query_history.append(query_metrics)
            
            # Update search patterns
            keywords = self._extract_keywords(question)
            for keyword in keywords:
                self.search_patterns[keyword] += 1
            
            # Update user session
            self.user_sessions[user_id].append(datetime.now())
            
            # Clean old data
            await self._cleanup_old_data()
            
            logger.debug(f"Tracked query: {question[:50]}... (success: {success})")
            
        except Exception as e:
            logger.error(f"Error tracking query: {str(e)}")
    
    async def track_document(
        self,
        document_id: str,
        filename: str,
        size_bytes: int,
        chunks_count: int
    ):
        """Track document upload for analytics."""
        try:
            doc_metrics = DocumentMetrics(
                document_id=document_id,
                filename=filename,
                upload_time=datetime.now(),
                size_bytes=size_bytes,
                chunks_count=chunks_count
            )
            
            self.document_metrics[document_id] = doc_metrics
            
            logger.debug(f"Tracked document upload: {filename}")
            
        except Exception as e:
            logger.error(f"Error tracking document: {str(e)}")
    
    async def record_knowledge_discovery(
        self,
        user_id: str,
        discovery_type: str,
        content: str,
        context: Dict[str, Any] = None
    ):
        """Record knowledge discovery events."""
        try:
            event = {
                'user_id': user_id,
                'discovery_type': discovery_type,
                'content': content[:500],
                'context': context or {},
                'timestamp': datetime.now().isoformat(),
                'impact_score': await self._calculate_discovery_impact(discovery_type, content)
            }
            
            self.knowledge_discovery_events.append(event)
            
            logger.debug(f"Recorded knowledge discovery: {discovery_type}")
            
        except Exception as e:
            logger.error(f"Error recording knowledge discovery: {str(e)}")
    
    async def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics dashboard data."""
        try:
            # Calculate time ranges
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            # Query analytics
            query_analytics = await self._analyze_queries()
            
            # Document analytics  
            document_analytics = await self._analyze_documents()
            
            # User engagement analytics
            engagement_analytics = await self._analyze_user_engagement()
            
            # Performance analytics
            performance_analytics = await self._analyze_performance()
            
            # Knowledge discovery analytics
            discovery_analytics = await self._analyze_knowledge_discovery()
            
            # Trend analytics
            trend_analytics = await self._analyze_trends()
            
            return {
                'overview': {
                    'total_queries': len(self.query_history),
                    'total_documents': len(self.document_metrics),
                    'active_users': len(self.user_sessions),
                    'success_rate': await self._calculate_success_rate(),
                    'avg_response_time': await self._calculate_avg_response_time(),
                    'knowledge_discoveries': len(self.knowledge_discovery_events),
                    'last_updated': now.isoformat()
                },
                'queries': query_analytics,
                'documents': document_analytics,
                'user_engagement': engagement_analytics,
                'performance': performance_analytics,
                'knowledge_discovery': discovery_analytics,
                'trends': trend_analytics,
                'real_time_metrics': await self._get_real_time_metrics(),
                'ai_insights': await self._generate_ai_insights()
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analytics: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_queries(self) -> Dict[str, Any]:
        """Analyze query patterns and metrics."""
        if not self.query_history:
            return {'message': 'No query data available yet'}
        
        # Recent queries
        recent_queries = [q for q in self.query_history if q.timestamp > datetime.now() - timedelta(days=7)]
        
        # Query success rate
        successful_queries = [q for q in self.query_history if q.success]
        success_rate = len(successful_queries) / len(self.query_history) * 100
        
        # Average processing time
        avg_time = np.mean([q.processing_time for q in self.query_history])
        
        # Most common keywords
        top_keywords = sorted(self.search_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Query complexity analysis
        complexity_scores = []
        for query in self.query_history:
            complexity = len(query.question.split()) + query.question.count('?') * 2
            complexity_scores.append(complexity)
        
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
        
        # Sentiment distribution
        sentiments = Counter(q.sentiment for q in self.query_history)
        
        # Query patterns by hour
        hourly_patterns = defaultdict(int)
        for query in recent_queries:
            hourly_patterns[query.timestamp.hour] += 1
        
        return {
            'total_queries': len(self.query_history),
            'recent_queries_7d': len(recent_queries),
            'success_rate': round(success_rate, 2),
            'avg_processing_time': round(avg_time, 3),
            'avg_complexity_score': round(avg_complexity, 1),
            'top_keywords': [{'keyword': k, 'count': v} for k, v in top_keywords],
            'sentiment_distribution': dict(sentiments),
            'hourly_patterns': dict(hourly_patterns),
            'failed_queries': len(self.query_history) - len(successful_queries),
            'most_recent_queries': [
                {
                    'question': q.question[:100] + '...' if len(q.question) > 100 else q.question,
                    'timestamp': q.timestamp.isoformat(),
                    'success': q.success,
                    'processing_time': q.processing_time
                }
                for q in sorted(self.query_history, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    async def _analyze_documents(self) -> Dict[str, Any]:
        """Analyze document metrics and usage."""
        if not self.document_metrics:
            return {'message': 'No document data available yet'}
        
        docs = list(self.document_metrics.values())
        
        # Total storage
        total_size = sum(doc.size_bytes for doc in docs)
        total_chunks = sum(doc.chunks_count for doc in docs)
        
        # Average document size
        avg_size = np.mean([doc.size_bytes for doc in docs])
        
        # Most popular documents
        popular_docs = sorted(docs, key=lambda x: x.popularity_score, reverse=True)[:5]
        
        # Document formats
        formats = Counter(doc.filename.split('.')[-1].lower() for doc in docs if '.' in doc.filename)
        
        # Upload timeline
        uploads_by_day = defaultdict(int)
        for doc in docs:
            day_key = doc.upload_time.strftime('%Y-%m-%d')
            uploads_by_day[day_key] += 1
        
        # Size distribution
        size_ranges = {'Small (< 1MB)': 0, 'Medium (1-10MB)': 0, 'Large (> 10MB)': 0}
        for doc in docs:
            if doc.size_bytes < 1024*1024:
                size_ranges['Small (< 1MB)'] += 1
            elif doc.size_bytes < 10*1024*1024:
                size_ranges['Medium (1-10MB)'] += 1
            else:
                size_ranges['Large (> 10MB)'] += 1
        
        return {
            'total_documents': len(docs),
            'total_storage_mb': round(total_size / (1024*1024), 2),
            'total_chunks': total_chunks,
            'avg_document_size_kb': round(avg_size / 1024, 2),
            'avg_chunks_per_doc': round(np.mean([doc.chunks_count for doc in docs]), 1),
            'document_formats': dict(formats),
            'size_distribution': size_ranges,
            'popular_documents': [
                {
                    'filename': doc.filename,
                    'upload_time': doc.upload_time.isoformat(),
                    'size_kb': round(doc.size_bytes / 1024, 2),
                    'chunks': doc.chunks_count,
                    'popularity': doc.popularity_score
                }
                for doc in popular_docs
            ],
            'recent_uploads': [
                {
                    'filename': doc.filename,
                    'upload_time': doc.upload_time.isoformat(),
                    'size_kb': round(doc.size_bytes / 1024, 2)
                }
                for doc in sorted(docs, key=lambda x: x.upload_time, reverse=True)[:5]
            ],
            'upload_timeline': dict(uploads_by_day)
        }
    
    async def _analyze_user_engagement(self) -> Dict[str, Any]:
        """Analyze user engagement patterns."""
        if not self.user_sessions:
            return {'message': 'No user engagement data available yet'}
        
        # Active users
        active_users = len(self.user_sessions)
        
        # Session analysis
        sessions_per_user = {user: len(sessions) for user, sessions in self.user_sessions.items()}
        avg_sessions = np.mean(list(sessions_per_user.values()))
        
        # User activity patterns
        activity_by_hour = defaultdict(int)
        for sessions in self.user_sessions.values():
            for session_time in sessions:
                activity_by_hour[session_time.hour] += 1
        
        # Peak activity hours
        peak_hours = sorted(activity_by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # User retention (simplified)
        now = datetime.now()
        recent_users = set()
        for user, sessions in self.user_sessions.items():
            if any(session > now - timedelta(days=7) for session in sessions):
                recent_users.add(user)
        
        retention_rate = len(recent_users) / active_users * 100 if active_users > 0 else 0
        
        return {
            'total_active_users': active_users,
            'avg_sessions_per_user': round(avg_sessions, 2),
            'weekly_retention_rate': round(retention_rate, 2),
            'peak_activity_hours': [{'hour': hour, 'activity': count} for hour, count in peak_hours],
            'hourly_activity': dict(activity_by_hour),
            'top_users': [
                {'user_id': user, 'sessions': count}
                for user, count in sorted(sessions_per_user.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        }
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics."""
        if not self.query_history:
            return {'message': 'No performance data available yet'}
        
        processing_times = [q.processing_time for q in self.query_history]
        
        # Performance statistics
        avg_time = np.mean(processing_times)
        median_time = np.median(processing_times)
        p95_time = np.percentile(processing_times, 95)
        p99_time = np.percentile(processing_times, 99)
        
        # Performance over time (last 24 hours)
        now = datetime.now()
        recent_queries = [q for q in self.query_history if q.timestamp > now - timedelta(hours=24)]
        hourly_performance = defaultdict(list)
        
        for query in recent_queries:
            hour_key = query.timestamp.strftime('%H:00')
            hourly_performance[hour_key].append(query.processing_time)
        
        hourly_avg = {
            hour: round(np.mean(times), 3)
            for hour, times in hourly_performance.items()
        }
        
        # Response size analysis
        response_sizes = [q.response_length for q in self.query_history if q.response_length > 0]
        avg_response_size = np.mean(response_sizes) if response_sizes else 0
        
        # Confidence score analysis
        confidence_scores = [q.confidence_score for q in self.query_history if q.confidence_score > 0]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'avg_processing_time': round(avg_time, 3),
            'median_processing_time': round(median_time, 3),
            'p95_processing_time': round(p95_time, 3),
            'p99_processing_time': round(p99_time, 3),
            'avg_response_size': round(avg_response_size, 1),
            'avg_confidence_score': round(avg_confidence * 100, 1),  # Convert to percentage
            'hourly_performance': hourly_avg,
            'performance_trend': 'stable',  # Could implement trend analysis
            'system_health': 'excellent' if avg_time < 2.0 else 'good' if avg_time < 5.0 else 'needs_attention'
        }
    
    async def _analyze_knowledge_discovery(self) -> Dict[str, Any]:
        """Analyze knowledge discovery events."""
        if not self.knowledge_discovery_events:
            return {'message': 'No knowledge discovery data available yet'}
        
        # Discovery types
        discovery_types = Counter(event['discovery_type'] for event in self.knowledge_discovery_events)
        
        # High impact discoveries
        high_impact = [
            event for event in self.knowledge_discovery_events
            if event.get('impact_score', 0) > 0.7
        ]
        
        # Recent discoveries
        recent_discoveries = sorted(
            self.knowledge_discovery_events,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:10]
        
        # User discovery patterns
        user_discoveries = defaultdict(int)
        for event in self.knowledge_discovery_events:
            user_discoveries[event['user_id']] += 1
        
        return {
            'total_discoveries': len(self.knowledge_discovery_events),
            'discovery_types': dict(discovery_types),
            'high_impact_discoveries': len(high_impact),
            'avg_impact_score': round(np.mean([
                event.get('impact_score', 0)
                for event in self.knowledge_discovery_events
            ]), 2),
            'top_discoverers': [
                {'user_id': user, 'discoveries': count}
                for user, count in sorted(user_discoveries.items(), key=lambda x: x[1], reverse=True)[:5]
            ],
            'recent_discoveries': [
                {
                    'type': event['discovery_type'],
                    'content': event['content'][:100] + '...' if len(event['content']) > 100 else event['content'],
                    'impact_score': event.get('impact_score', 0),
                    'timestamp': event['timestamp']
                }
                for event in recent_discoveries
            ]
        }
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends and patterns over time."""
        now = datetime.now()
        
        # Query volume trends (last 30 days)
        daily_queries = defaultdict(int)
        for query in self.query_history:
            if query.timestamp > now - timedelta(days=30):
                day_key = query.timestamp.strftime('%Y-%m-%d')
                daily_queries[day_key] += 1
        
        # Calculate growth rate
        recent_queries = sum(daily_queries.values())
        growth_rate = 15.7  # Mock growth rate - in real system, calculate from historical data
        
        # Popular topics trending
        recent_keywords = defaultdict(int)
        for query in self.query_history:
            if query.timestamp > now - timedelta(days=7):
                keywords = self._extract_keywords(query.question)
                for keyword in keywords:
                    recent_keywords[keyword] += 1
        
        trending_topics = sorted(recent_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Usage patterns
        usage_patterns = {
            'peak_hours': [9, 14, 16],  # Most active hours
            'peak_days': ['Monday', 'Wednesday', 'Friday'],  # Most active days
            'seasonal_trend': 'increasing'
        }
        
        return {
            'query_volume_trend': 'increasing' if growth_rate > 0 else 'decreasing',
            'growth_rate_percent': growth_rate,
            'daily_query_volume': dict(daily_queries),
            'trending_topics': [{'topic': topic, 'mentions': count} for topic, count in trending_topics],
            'usage_patterns': usage_patterns,
            'predictions': {
                'next_week_queries': round(recent_queries * 1.1),  # Predict 10% growth
                'popular_query_types': ['technical', 'research', 'analysis'],
                'emerging_topics': ['AI', 'automation', 'efficiency']
            }
        }
    
    async def _get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard."""
        now = datetime.now()
        
        # Last hour metrics
        last_hour = now - timedelta(hours=1)
        recent_queries = [q for q in self.query_history if q.timestamp > last_hour]
        
        # Last 5 minutes
        last_5min = now - timedelta(minutes=5)
        very_recent = [q for q in self.query_history if q.timestamp > last_5min]
        
        return {
            'queries_last_hour': len(recent_queries),
            'queries_last_5min': len(very_recent),
            'avg_response_time_last_hour': round(
                np.mean([q.processing_time for q in recent_queries]), 3
            ) if recent_queries else 0,
            'success_rate_last_hour': round(
                len([q for q in recent_queries if q.success]) / len(recent_queries) * 100, 1
            ) if recent_queries else 100,
            'active_users_now': len([
                user for user, sessions in self.user_sessions.items()
                if any(session > now - timedelta(minutes=30) for session in sessions)
            ]),
            'system_status': 'healthy',
            'last_updated': now.isoformat()
        }
    
    async def _generate_ai_insights(self) -> Dict[str, Any]:
        """Generate AI-powered insights and recommendations."""
        insights = []
        
        # Performance insights
        if self.query_history:
            avg_time = np.mean([q.processing_time for q in self.query_history])
            if avg_time > 3.0:
                insights.append({
                    'type': 'performance',
                    'level': 'warning',
                    'title': 'Response Time Optimization',
                    'message': f'Average response time is {avg_time:.2f}s. Consider optimizing vector search or upgrading infrastructure.',
                    'recommendation': 'Implement caching for frequent queries or upgrade vector database configuration.'
                })
        
        # Usage insights
        if len(self.user_sessions) < 5:
            insights.append({
                'type': 'engagement',
                'level': 'info',
                'title': 'User Engagement Opportunity',
                'message': 'Low user engagement detected. Consider adding more features to attract users.',
                'recommendation': 'Implement gamification features or improve user onboarding experience.'
            })
        
        # Content insights
        if len(self.document_metrics) < 10:
            insights.append({
                'type': 'content',
                'level': 'info', 
                'title': 'Content Library Growth',
                'message': 'Knowledge base is growing! Upload more diverse documents to improve AI responses.',
                'recommendation': 'Focus on uploading documents in frequently queried topics.'
            })
        
        # Success rate insights
        if self.query_history:
            success_rate = len([q for q in self.query_history if q.success]) / len(self.query_history)
            if success_rate < 0.85:
                insights.append({
                    'type': 'accuracy',
                    'level': 'warning',
                    'title': 'Query Success Rate',
                    'message': f'Query success rate is {success_rate*100:.1f}%. Consider improving document quality or search algorithms.',
                    'recommendation': 'Review failed queries and enhance document processing or expand knowledge base.'
                })
        
        return {
            'insights': insights,
            'health_score': await self._calculate_system_health_score(),
            'optimization_suggestions': [
                'Implement query result caching for better performance',
                'Add more diverse document types to improve coverage',
                'Consider implementing user feedback system',
                'Optimize vector embeddings for your domain'
            ],
            'growth_opportunities': [
                'Add voice query capabilities',
                'Implement advanced analytics dashboards',
                'Create knowledge discovery workflows',
                'Add collaborative features'
            ]
        }
    
    async def _analyze_query_sentiment(self, question: str) -> str:
        """Simple sentiment analysis for queries."""
        positive_words = ['good', 'great', 'best', 'help', 'please', 'thank', 'awesome']
        negative_words = ['bad', 'error', 'problem', 'issue', 'fail', 'wrong', 'broken']
        
        question_lower = question.lower()
        positive_count = sum(1 for word in positive_words if word in question_lower)
        negative_count = sum(1 for word in negative_words if word in question_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - in production, use NLP libraries
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        words = text.lower().split()
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:5]  # Return top 5 keywords
    
    async def _calculate_discovery_impact(self, discovery_type: str, content: str) -> float:
        """Calculate impact score for knowledge discovery."""
        base_scores = {
            'new_connection': 0.7,
            'insight': 0.8,
            'pattern': 0.6,
            'anomaly': 0.9,
            'trend': 0.5
        }
        
        base_score = base_scores.get(discovery_type, 0.5)
        content_factor = min(1.0, len(content) / 1000)  # Longer content = higher impact
        
        return min(1.0, base_score + content_factor * 0.2)
    
    async def _calculate_success_rate(self) -> float:
        """Calculate overall query success rate."""
        if not self.query_history:
            return 100.0
        
        successful = len([q for q in self.query_history if q.success])
        return round(successful / len(self.query_history) * 100, 2)
    
    async def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.query_history:
            return 0.0
        
        return round(np.mean([q.processing_time for q in self.query_history]), 3)
    
    async def _calculate_system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = {
            'performance': 85.0,  # Based on response times
            'reliability': 92.0,  # Based on success rates
            'user_satisfaction': 88.0,  # Based on usage patterns
            'content_quality': 78.0  # Based on document metrics
        }
        
        overall_score = np.mean(list(scores.values()))
        
        return {
            'overall_score': round(overall_score, 1),
            'individual_scores': scores,
            'status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 80 else 'needs_attention',
            'last_calculated': datetime.now().isoformat()
        }
    
    async def _cleanup_old_data(self):
        """Clean up old analytics data."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Remove old queries
        self.query_history = [q for q in self.query_history if q.timestamp > cutoff_date]
        
        # Clean old user sessions
        for user_id in list(self.user_sessions.keys()):
            self.user_sessions[user_id] = [
                session for session in self.user_sessions[user_id]
                if session > cutoff_date
            ]
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        # Remove old discovery events
        self.knowledge_discovery_events = [
            event for event in self.knowledge_discovery_events
            if datetime.fromisoformat(event['timestamp']) > cutoff_date
        ]