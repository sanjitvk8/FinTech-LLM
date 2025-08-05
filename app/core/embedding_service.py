import logging
import numpy as np
from typing import List, Dict, Any
import openai
from sentence_transformers import SentenceTransformer
import asyncio
import tiktoken
from .config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI and SentenceTransformers"""
    
    def __init__(self):
        self.has_openai_key = bool(settings.OPENAI_API_KEY)
        if self.has_openai_key:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self.openai_client = None
            self.encoding = None
            logger.warning("OpenAI API key not provided - OpenAI embeddings will not be available")
        
        # Always initialize SentenceTransformer as fallback
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}. Using dummy embeddings.")
            self.sentence_transformer = None
        
    async def generate_embeddings(self, texts: List[str], method: str = "auto") -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            # Auto-select method based on availability
            if method == "auto":
                if self.has_openai_key and self.openai_client:
                    method = "openai"
                elif self.sentence_transformer:
                    method = "sentence_transformer"
                else:
                    logger.warning("No embedding services available, using dummy embeddings")
                    return self._generate_dummy_embeddings(texts)
            
            if method == "openai":
                if not self.has_openai_key:
                    logger.warning("OpenAI API key not available, falling back to SentenceTransformer")
                    return await self._generate_sentence_transformer_embeddings(texts)
                try:
                    return await self._generate_openai_embeddings(texts)
                except Exception as openai_error:
                    logger.error(f"OpenAI embeddings failed: {str(openai_error)}")
                    if "429" in str(openai_error) or "quota" in str(openai_error).lower():
                        logger.info("OpenAI quota exceeded, falling back to SentenceTransformer")
                        if self.sentence_transformer:
                            return await self._generate_sentence_transformer_embeddings(texts)
                        else:
                            logger.warning("SentenceTransformer not available, using dummy embeddings")
                            return self._generate_dummy_embeddings(texts)
                    else:
                        raise openai_error
            elif method == "sentence_transformer":
                if not self.sentence_transformer:
                    logger.warning("SentenceTransformer not available, generating dummy embeddings")
                    return self._generate_dummy_embeddings(texts)
                return await self._generate_sentence_transformer_embeddings(texts)
            else:
                raise ValueError(f"Unsupported embedding method: {method}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}. Falling back to dummy embeddings.")
            return self._generate_dummy_embeddings(texts)
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Process in batches to avoid rate limits
            batch_size = 20
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate texts if they're too long
                truncated_batch = []
                for text in batch:
                    tokens = self.encoding.encode(text)
                    if len(tokens) > 8000:  # Safe limit for embedding models
                        truncated_tokens = tokens[:8000]
                        truncated_text = self.encoding.decode(truncated_tokens)
                        truncated_batch.append(truncated_text)
                    else:
                        truncated_batch.append(text)
                
                response = await asyncio.to_thread(
                    self.openai_client.embeddings.create,
                    input=truncated_batch,
                    model=settings.EMBEDDING_MODEL
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error with OpenAI embeddings: {str(e)}")
            raise
    
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformers"""
        try:
            embeddings = await asyncio.to_thread(
                self.sentence_transformer.encode,
                texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error with SentenceTransformer embeddings: {str(e)}")
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def find_most_similar(self, query_embedding: List[float], 
                              chunk_embeddings: List[List[float]], 
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar chunks to query"""
        try:
            similarities = []
            
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = self.calculate_similarity(query_embedding, chunk_embedding)
                similarities.append({
                    'chunk_index': i,
                    'similarity_score': similarity
                })
            
            # Sort by similarity score and return top_k
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            return []
    
    def _generate_dummy_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dummy embeddings for testing when no embedding service is available"""
        dimension = 384  # Match sentence transformer dimension
        embeddings = []
        for text in texts:
            # Generate deterministic dummy embedding based on text hash
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.normal(0, 1, dimension).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def get_embedding_dimension(self, method: str = "openai") -> int:
        """Get the dimension of embeddings for the specified method"""
        if method == "openai":
            return 1536  # text-embedding-3-small dimension
        elif method == "sentence_transformer":
            return 384   # all-MiniLM-L6-v2 dimension
        else:
            return 384  # default to sentence transformer dimension

