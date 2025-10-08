"""Professional implementation"""

import asyncio
import json
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime
from loguru import logger
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("SpaCy not available. Using simplified entity extraction.")
    SPACY_AVAILABLE = False
    spacy = None
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np

from backend.app.config import settings

class KnowledgeGraphService:
    """Service for building and analyzing knowledge graphs from documents."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_cache = {}
        self.topic_clusters = {}
        self.relationship_patterns = [
            r'\b(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+)',
            r'\b(\w+)\s+(?:has|have|had)\s+(?:a|an|the)?\s*(\w+)',
            r'\b(\w+)\s+(?:uses|use|used)\s+(?:a|an|the)?\s*(\w+)',
            r'\b(\w+)\s+(?:contains|contain|contained)\s+(?:a|an|the)?\s*(\w+)',
            r'\b(\w+)\s+(?:causes|cause|caused)\s+(?:a|an|the)?\s*(\w+)',
        ]
        
        # Initialize NLP model (you might need to download this first)
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
    
    async def analyze_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze documents and build knowledge graph.
        
        Args:
            documents: List of document data with content and metadata
            
        Returns:
            Knowledge graph data with nodes, edges, and analytics
        """
        logger.info(f"Analyzing {len(documents)} documents for knowledge graph")
        
        # Clear previous graph
        self.graph.clear()
        self.entity_cache = {}
        
        # Process each document
        all_entities = []
        all_topics = []
        document_topics = {}
        
        for doc in documents:
            doc_id = doc.get('document_id', doc.get('id', 'unknown'))
            content = doc.get('content', doc.get('text_content', ''))
            filename = doc.get('filename', f'document_{doc_id}')
            
            # Extract entities and relationships
            entities = await self._extract_entities(content, doc_id)
            topics = await self._extract_topics(content)
            relationships = await self._extract_relationships(content)
            
            # Store for clustering
            all_entities.extend(entities)
            all_topics.extend(topics)
            document_topics[doc_id] = {
                'filename': filename,
                'topics': topics,
                'entities': entities,
                'relationships': relationships
            }
            
            # Add document node
            self.graph.add_node(doc_id, 
                               type='document',
                               label=filename,
                               size=20,
                               color='#ff6b6b',
                               entities=len(entities),
                               topics=len(topics))
            
            # Add entity nodes and relationships
            for entity in entities:
                entity_id = f"entity_{entity['text'].lower().replace(' ', '_')}"
                
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id,
                                      type='entity',
                                      label=entity['text'],
                                      entity_type=entity['label'],
                                      size=10,
                                      color=self._get_entity_color(entity['label']),
                                      frequency=1)
                else:
                    # Increment frequency
                    self.graph.nodes[entity_id]['frequency'] += 1
                    self.graph.nodes[entity_id]['size'] = min(25, 10 + self.graph.nodes[entity_id]['frequency'])
                
                # Connect entity to document
                self.graph.add_edge(doc_id, entity_id, 
                                  type='contains',
                                  weight=entity['confidence'])
            
            # Add topic nodes
            for i, topic in enumerate(topics):
                topic_id = f"topic_{topic.lower().replace(' ', '_')}"
                
                if not self.graph.has_node(topic_id):
                    self.graph.add_node(topic_id,
                                      type='topic',
                                      label=topic,
                                      size=15,
                                      color='#4ecdc4',
                                      frequency=1)
                else:
                    self.graph.nodes[topic_id]['frequency'] += 1
                    self.graph.nodes[topic_id]['size'] = min(30, 15 + self.graph.nodes[topic_id]['frequency'] * 2)
                
                # Connect topic to document
                self.graph.add_edge(doc_id, topic_id,
                                  type='discusses',
                                  weight=1.0)
            
            # Add relationship edges
            for rel in relationships:
                entity1_id = f"entity_{rel['entity1'].lower().replace(' ', '_')}"
                entity2_id = f"entity_{rel['entity2'].lower().replace(' ', '_')}"
                
                if self.graph.has_node(entity1_id) and self.graph.has_node(entity2_id):
                    if not self.graph.has_edge(entity1_id, entity2_id):
                        self.graph.add_edge(entity1_id, entity2_id,
                                          type='relationship',
                                          relation=rel['relation'],
                                          weight=rel['confidence'])
        
        # Perform clustering analysis
        clusters = await self._cluster_topics(all_topics, document_topics)
        
        # Calculate graph metrics
        metrics = await self._calculate_graph_metrics()
        
        # Convert graph to vis.js format
        graph_data = await self._export_to_visjs()
        
        return {
            'graph': graph_data,
            'clusters': clusters,
            'metrics': metrics,
            'summary': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'unique_entities': len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'entity']),
                'unique_topics': len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'topic']),
                'documents': len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'document'])
            },
            'generated_at': datetime.now().isoformat()
        }
    
    async def _extract_entities(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using SpaCy."""
        if not self.nlp:
            return []
        
        # Cache entities per document
        if doc_id in self.entity_cache:
            return self.entity_cache[doc_id]
        
        doc = self.nlp(text[:10000])  # Limit text length for performance
        entities = []
        
        for ent in doc.ents:
            if len(ent.text.strip()) > 2 and ent.label_ not in ['CARDINAL', 'ORDINAL', 'QUANTITY']:
                entities.append({
                    'text': ent.text.strip(),
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8  # Default confidence for SpaCy entities
                })
        
        # Remove duplicates and sort by confidence
        unique_entities = []
        seen_texts = set()
        for ent in entities:
            if ent['text'].lower() not in seen_texts:
                seen_texts.add(ent['text'].lower())
                unique_entities.append(ent)
        
        self.entity_cache[doc_id] = unique_entities[:50]  # Limit to top 50 entities
        return self.entity_cache[doc_id]
    
    async def _extract_topics(self, text: str, max_topics: int = 10) -> List[str]:
        """Extract key topics from text using TF-IDF."""
        # Simple topic extraction using TF-IDF
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-max_topics:][::-1]
            
            topics = [feature_names[i] for i in top_indices if scores[i] > 0.1]
            return topics[:max_topics]
            
        except Exception as e:
            logger.warning(f"Error extracting topics: {str(e)}")
            return []
    
    async def _extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities using pattern matching."""
        relationships = []
        
        for pattern in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity1, entity2 = match.groups()
                if len(entity1) > 2 and len(entity2) > 2:
                    relationships.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'relation': 'related_to',  # Simplified relation type
                        'confidence': 0.6,
                        'pattern': pattern
                    })
        
        return relationships[:20]  # Limit relationships
    
    async def _cluster_topics(self, all_topics: List[str], document_topics: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster topics to find thematic groups."""
        if len(all_topics) < 3:
            return {'clusters': [], 'method': 'insufficient_data'}
        
        try:
            # Create topic frequency matrix
            topic_freq = Counter(all_topics)
            topics = list(topic_freq.keys())
            
            if len(topics) < 3:
                return {'clusters': [], 'method': 'insufficient_unique_topics'}
            
            # Simple clustering based on co-occurrence
            clusters = defaultdict(list)
            
            # Group documents by similar topics
            doc_topic_vectors = {}
            for doc_id, doc_data in document_topics.items():
                vector = [1 if topic in doc_data['topics'] else 0 for topic in topics]
                doc_topic_vectors[doc_id] = vector
            
            # Cluster documents
            vectors = list(doc_topic_vectors.values())
            if len(vectors) > 1:
                n_clusters = min(5, len(vectors))  # Max 5 clusters
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(vectors)
                    
                    for i, doc_id in enumerate(doc_topic_vectors.keys()):
                        cluster_id = int(cluster_labels[i])
                        clusters[f"cluster_{cluster_id}"].append({
                            'document_id': doc_id,
                            'filename': document_topics[doc_id]['filename'],
                            'topics': document_topics[doc_id]['topics'][:5]
                        })
                    
                    return {
                        'clusters': dict(clusters),
                        'method': 'kmeans',
                        'n_clusters': n_clusters
                    }
                    
                except Exception as e:
                    logger.warning(f"Clustering failed: {str(e)}")
            
            return {'clusters': [], 'method': 'clustering_failed'}
            
        except Exception as e:
            logger.error(f"Error in topic clustering: {str(e)}")
            return {'clusters': [], 'method': 'error'}
    
    async def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate various graph metrics and statistics."""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            # Basic metrics
            metrics = {
                'density': nx.density(self.graph),
                'number_of_connected_components': nx.number_connected_components(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
            }
            
            # Centrality measures (for smaller graphs)
            if self.graph.number_of_nodes() < 1000:
                centrality = nx.degree_centrality(self.graph)
                betweenness = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
                
                # Find most central nodes
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
                
                metrics.update({
                    'most_central_nodes': [(node, self.graph.nodes[node].get('label', node), score) 
                                         for node, score in top_central],
                    'most_between_nodes': [(node, self.graph.nodes[node].get('label', node), score) 
                                         for node, score in top_betweenness]
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {str(e)}")
            return {'error': str(e)}
    
    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity type."""
        color_map = {
            'PERSON': '#ff9999',
            'ORG': '#66b3ff',
            'GPE': '#99ff99',  # Geopolitical entity
            'MONEY': '#ffcc99',
            'DATE': '#ff99cc',
            'TIME': '#ccffcc',
            'PRODUCT': '#ffccff',
            'EVENT': '#ccccff',
            'WORK_OF_ART': '#ffffcc',
            'LANGUAGE': '#ffcccc',
            'NORP': '#ccffff',  # Nationalities, political groups
        }
        return color_map.get(entity_type, '#cccccc')
    
    async def _export_to_visjs(self) -> Dict[str, Any]:
        """Export graph to vis.js compatible format."""
        nodes = []
        edges = []
        
        # Convert nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            nodes.append({
                'id': node_id,
                'label': node_data.get('label', node_id),
                'size': node_data.get('size', 10),
                'color': node_data.get('color', '#cccccc'),
                'type': node_data.get('type', 'unknown'),
                'title': self._create_node_tooltip(node_id, node_data),
                'physics': True,
                'borderWidth': 2,
                'borderColor': '#2c3e50'
            })
        
        # Convert edges
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            edges.append({
                'from': source,
                'to': target,
                'width': min(10, max(1, data.get('weight', 1) * 5)),
                'color': self._get_edge_color(data.get('type', 'unknown')),
                'title': f"{data.get('type', 'connection')} (weight: {data.get('weight', 1):.2f})",
                'arrows': {'to': {'enabled': True, 'scaleFactor': 0.5}},
                'smooth': {'type': 'continuous', 'roundness': 0.5}
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'physics': {
                'enabled': True,
                'barnesHut': {
                    'gravitationalConstant': -8000,
                    'centralGravity': 0.3,
                    'springLength': 95,
                    'springConstant': 0.04,
                    'damping': 0.09
                }
            },
            'layout': {
                'improvedLayout': True
            }
        }
    
    def _create_node_tooltip(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Create tooltip for node."""
        tooltip_parts = [f"<b>{node_data.get('label', node_id)}</b>"]
        
        if node_data.get('type') == 'document':
            tooltip_parts.append(f"📄 Document")
            tooltip_parts.append(f"Entities: {node_data.get('entities', 0)}")
            tooltip_parts.append(f"Topics: {node_data.get('topics', 0)}")
        elif node_data.get('type') == 'entity':
            tooltip_parts.append(f"🔖 {node_data.get('entity_type', 'Entity')}")
            tooltip_parts.append(f"Frequency: {node_data.get('frequency', 1)}")
        elif node_data.get('type') == 'topic':
            tooltip_parts.append(f"💭 Topic")
            tooltip_parts.append(f"Frequency: {node_data.get('frequency', 1)}")
        
        return "<br>".join(tooltip_parts)
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for edge type."""
        color_map = {
            'contains': '#3498db',
            'discusses': '#2ecc71',
            'relationship': '#e74c3c',
            'related_to': '#f39c12'
        }
        return color_map.get(edge_type, '#95a5a6')
    
    async def get_node_neighborhood(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get neighborhood of a specific node."""
        if not self.graph.has_node(node_id):
            return {'error': 'Node not found'}
        
        # Get neighbors within specified depth
        neighbors = set([node_id])
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
            current_level = next_level - neighbors
            neighbors.update(current_level)
        
        # Create subgraph
        subgraph = self.graph.subgraph(neighbors)
        
        # Convert to vis.js format
        return await self._export_subgraph_to_visjs(subgraph)
    
    async def _export_subgraph_to_visjs(self, subgraph) -> Dict[str, Any]:
        """Export subgraph to vis.js format."""
        nodes = []
        edges = []
        
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            nodes.append({
                'id': node_id,
                'label': node_data.get('label', node_id),
                'size': node_data.get('size', 10),
                'color': node_data.get('color', '#cccccc'),
                'type': node_data.get('type', 'unknown')
            })
        
        for edge in subgraph.edges(data=True):
            source, target, data = edge
            edges.append({
                'from': source,
                'to': target,
                'width': min(10, max(1, data.get('weight', 1) * 5)),
                'color': self._get_edge_color(data.get('type', 'unknown'))
            })
        
        return {'nodes': nodes, 'edges': edges}