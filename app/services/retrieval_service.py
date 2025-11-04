import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from ..core.embedding_service import EmbeddingService
from ..utils.vector_store import VectorStore
from ..core.config import settings
import numpy as np

logger = logging.getLogger(__name__)

class RetrievalService:
    """Service for intelligent document retrieval and ranking"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        
    async def retrieve_relevant_chunks(self, query: str, vector_store: VectorStore,
                                     top_k: int = None, 
                                     rerank: bool = True) -> List[Dict[str, Any]]:
        """Retrieve and optionally rerank relevant document chunks"""
        try:
            if top_k is None:
                top_k = settings.MAX_RETRIEVED_CHUNKS
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embeddings([query])
            
            # Initial retrieval
            initial_results = vector_store.search(
                query_embedding[0],
                k=min(top_k * 2, 50),  # Retrieve more for reranking
                threshold=settings.SIMILARITY_THRESHOLD * 0.8  # Lower threshold for initial retrieval
            )
            
            if not initial_results:
                logger.warning("No relevant chunks found for query")
                return []
            
            # Rerank if requested
            if rerank and len(initial_results) > top_k:
                reranked_results = await self._rerank_results(query, initial_results, top_k)
                return reranked_results[:top_k]
            
            return initial_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            raise
    
    async def _rerank_results(self, query: str, initial_results: List[Dict[str, Any]], 
                            top_k: int) -> List[Dict[str, Any]]:
        """Rerank initial results using advanced scoring"""
        try:
            # Extract chunk contents
            chunk_contents = []
            for result in initial_results:
                chunk_content = result.get('chunk', {}).get('content', '')
                chunk_contents.append(chunk_content)
            
            # Calculate advanced relevance scores
            relevance_scores = await self._calculate_relevance_scores(query, chunk_contents)
            
            # Combine similarity and relevance scores
            for i, result in enumerate(initial_results):
                similarity_score = result.get('similarity_score', 0.0)
                relevance_score = relevance_scores[i] if i < len(relevance_scores) else 0.0
                
                # Weighted combination (60% similarity, 40% relevance)
                combined_score = 0.6 * similarity_score + 0.4 * relevance_score
                result['relevance_score'] = relevance_score
                result['combined_score'] = combined_score
            
            # Sort by combined score
            reranked_results = sorted(
                initial_results, 
                key=lambda x: x.get('combined_score', 0.0), 
                reverse=True
            )
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error reranking results: {str(e)}")
            return initial_results
    
    async def _calculate_relevance_scores(self, query: str, 
                                        chunk_contents: List[str]) -> List[float]:
        """Calculate relevance scores using various techniques"""
        try:
            scores = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for content in chunk_contents:
                content_lower = content.lower()
                content_words = set(content_lower.split())
                
                # 1. Keyword overlap score
                keyword_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
                
                # 2. Phrase matching score
                phrase_matches = 0
                for word in query_words:
                    if word in content_lower:
                        phrase_matches += 1
                phrase_score = phrase_matches / len(query_words) if query_words else 0
                
                # 3. Length penalty (prefer neither too short nor too long chunks)
                optimal_length = 500  # characters
                length_penalty = 1.0 - abs(len(content) - optimal_length) / (2 * optimal_length)
                length_penalty = max(0.1, length_penalty)  # Minimum penalty
                
                # 4. Position bonus (if chunk metadata contains position info)
                position_bonus = 1.0  # Default, can be enhanced with actual position data
                
                # Combine scores
                combined_relevance = (
                    0.4 * keyword_overlap +
                    0.3 * phrase_score +
                    0.2 * length_penalty +
                    0.1 * position_bonus
                )
                
                scores.append(combined_relevance)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating relevance scores: {str(e)}")
            return [0.5] * len(chunk_contents)  # Default scores
    
    async def multi_query_retrieval(self, queries: List[str], vector_store: VectorStore,
                                  top_k_per_query: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve relevant chunks for multiple queries"""
        try:
            results = {}
            
            # Process queries concurrently
            tasks = []
            for query in queries:
                task = asyncio.create_task(
                    self.retrieve_relevant_chunks(query, vector_store, top_k_per_query),
                    name=f"retrieve_{hash(query)}"
                )
                tasks.append((query, task))
            
            # Collect results
            for query, task in tasks:
                try:
                    query_results = await task
                    results[query] = query_results
                except Exception as e:
                    logger.error(f"Error retrieving for query '{query}': {str(e)}")
                    results[query] = []
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-query retrieval: {str(e)}")
            raise
    
    def filter_results_by_criteria(self, results: List[Dict[str, Any]], 
                                 criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter results based on various criteria"""
        try:
            filtered_results = []
            
            for result in results:
                # Check minimum similarity score
                if criteria.get('min_similarity', 0.0) > result.get('similarity_score', 0.0):
                    continue
                
                # Check maximum results per document
                # (This would need document tracking in metadata)
                
                # Check content length requirements
                chunk_content = result.get('chunk', {}).get('content', '')
                min_length = criteria.get('min_content_length', 0)
                max_length = criteria.get('max_content_length', float('inf'))
                
                if not (min_length <= len(chunk_content) <= max_length):
                    continue
                
                # Check for required keywords
                required_keywords = criteria.get('required_keywords', [])
                if required_keywords:
                    content_lower = chunk_content.lower()
                    if not all(keyword.lower() in content_lower for keyword in required_keywords):
                        continue
                
                # Check for excluded keywords
                excluded_keywords = criteria.get('excluded_keywords', [])
                if excluded_keywords:
                    content_lower = chunk_content.lower()
                    if any(keyword.lower() in content_lower for keyword in excluded_keywords):
                        continue
                
                filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error filtering results: {str(e)}")
            return results
    
    def deduplicate_results(self, results: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar chunks"""
        try:
            if not results:
                return results
            
            deduplicated = []
            seen_contents = []
            
            for result in results:
                chunk_content = result.get('chunk', {}).get('content', '')
                
                # Check for exact duplicates
                if chunk_content in seen_contents:
                    continue
                
                # Check for high similarity with existing results
                is_duplicate = False
                for seen_content in seen_contents:
                    similarity = self._calculate_text_similarity(chunk_content, seen_content)
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    deduplicated.append(result)
                    seen_contents.append(chunk_content)
            
            logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error deduplicating results: {str(e)}")
            return results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using Jaccard similarity"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0
    
    def get_retrieval_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about retrieval results"""
        try:
            if not results:
                return {
                    'total_results': 0,
                    'avg_similarity': 0.0,
                    'max_similarity': 0.0,
                    'min_similarity': 0.0
                }
            
            similarities = [r.get('similarity_score', 0.0) for r in results]
            
            return {
                'total_results': len(results),
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                'std_similarity': np.std(similarities),
                'results_above_threshold': len([s for s in similarities if s >= settings.SIMILARITY_THRESHOLD])
            }
            
        except Exception as e:
            logger.error(f"Error calculating retrieval stats: {str(e)}")
            return {'error': str(e)}