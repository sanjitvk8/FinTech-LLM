import logging
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store for document chunks"""
    
    def __init__(self, dimension: int = 1536, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.chunk_contents = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info(f"Initialized FAISS index: {self.index_type}, dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise
    
    def add_vectors(self, embeddings: List[List[float]], 
                   chunks: List[Dict[str, Any]], 
                   metadata: List[Dict[str, Any]]) -> None:
        """Add vectors to the index"""
        try:
            if len(embeddings) != len(chunks) or len(embeddings) != len(metadata):
                raise ValueError("Embeddings, chunks, and metadata must have the same length")
            
            if not embeddings:
                logger.warning("No embeddings to add")
                return
            
            # Convert to numpy array and normalize for cosine similarity
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Check if dimensions match the index
            if vectors.shape[1] != self.dimension:
                logger.warning(f"Embedding dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}. Reinitializing index.")
                self.dimension = vectors.shape[1]
                self._initialize_index()
            
            faiss.normalize_L2(vectors)
            
            # Add to index
            self.index.add(vectors)
            
            # Store metadata and content
            self.metadata.extend(metadata)
            self.chunk_contents.extend(chunks)
            
            logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error adding vectors to index: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 10, 
               threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            if self.index.ntotal == 0:
                logger.warning("Index is empty")
                return []
            
            # Normalize query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                if threshold and score < threshold:
                    continue
                
                results.append({
                    'chunk': self.chunk_contents[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(score),
                    'rank': i + 1
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []
    
    def save_index(self, filepath: str) -> None:
        """Save the index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata and chunks
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'chunk_contents': self.chunk_contents,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f)
            
            logger.info(f"Saved index to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, filepath: str) -> None:
        """Load the index and metadata from disk"""
        try:
            # Load FAISS index
            if not os.path.exists(f"{filepath}.faiss"):
                raise FileNotFoundError(f"Index file not found: {filepath}.faiss")
            
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata and chunks
            with open(f"{filepath}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.chunk_contents = data['chunk_contents']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
            
            logger.info(f"Loaded index from {filepath}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.metadata),
            'chunks_count': len(self.chunk_contents)
        }
    
    def clear(self) -> None:
        """Clear the index and metadata"""
        self._initialize_index()
        self.metadata = []
        self.chunk_contents = []
        logger.info("Cleared vector store")

