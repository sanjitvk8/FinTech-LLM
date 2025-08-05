import logging
import asyncio
from typing import List, Dict, Any, Optional
from ..core.document_processor import DocumentProcessor
from ..core.embedding_service import EmbeddingService
from ..utils.vector_store import VectorStore
from ..core.config import settings

logger = logging.getLogger(__name__)

class DocumentService:
    """Service for managing document processing workflows"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.processed_documents = {}  # Cache for processed documents
        
    async def process_and_index_document(self, document_url: str, 
                                       document_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a document and create searchable index"""
        try:
            # Use URL as document_id if not provided
            if not document_id:
                document_id = self._generate_document_id(document_url)
            
            # Check if already processed
            if document_id in self.processed_documents:
                logger.info(f"Document {document_id} already processed, using cache")
                return self.processed_documents[document_id]
            
            # Process document
            logger.info(f"Processing document: {document_url}")
            document_data = await self.document_processor.process_document_from_url(document_url)
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            chunk_texts = [chunk['content'] for chunk in document_data['chunks']]
            embeddings = await self.embedding_service.generate_embeddings(chunk_texts)
            
            # Create vector store
            vector_store = VectorStore(dimension=1536)
            
            # Prepare metadata
            metadata = []
            for i, chunk in enumerate(document_data['chunks']):
                metadata.append({
                    'document_id': document_id,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_index': i,
                    'source_url': document_url,
                    'size': chunk['size'],
                    'metadata': chunk.get('metadata', {})
                })
            
            # Add to vector store
            vector_store.add_vectors(embeddings, document_data['chunks'], metadata)
            
            # Cache results
            result = {
                'document_id': document_id,
                'document_data': document_data,
                'vector_store': vector_store,
                'embeddings': embeddings,
                'metadata': metadata,
                'processing_stats': {
                    'total_chunks': len(document_data['chunks']),
                    'total_embeddings': len(embeddings),
                    'content_length': len(document_data['content'])
                }
            }
            
            self.processed_documents[document_id] = result
            
            logger.info(f"Successfully processed document {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_url}: {str(e)}")
            raise
    
    async def search_document(self, document_id: str, query: str, 
                            top_k: int = 10) -> List[Dict[str, Any]]:
        """Search within a specific document"""
        try:
            if document_id not in self.processed_documents:
                raise ValueError(f"Document {document_id} not found or not processed")
            
            doc_data = self.processed_documents[document_id]
            vector_store = doc_data['vector_store']
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embeddings([query])
            
            # Search
            results = vector_store.search(
                query_embedding[0], 
                k=top_k,
                threshold=settings.SIMILARITY_THRESHOLD
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching document {document_id}: {str(e)}")
            raise
    
    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get summary information about a processed document"""
        try:
            if document_id not in self.processed_documents:
                raise ValueError(f"Document {document_id} not found")
            
            doc_data = self.processed_documents[document_id]
            
            return {
                'document_id': document_id,
                'processing_stats': doc_data['processing_stats'],
                'vector_store_stats': doc_data['vector_store'].get_stats(),
                'content_preview': doc_data['document_data']['content'][:500] + "...",
                'first_chunk_preview': doc_data['document_data']['chunks'][0]['content'][:200] + "..." if doc_data['document_data']['chunks'] else ""
            }
            
        except Exception as e:
            logger.error(f"Error getting document summary {document_id}: {str(e)}")
            raise
    
    def _generate_document_id(self, document_url: str) -> str:
        """Generate a unique document ID from URL"""
        import hashlib
        return hashlib.md5(document_url.encode()).hexdigest()[:16]
    
    def get_processed_documents(self) -> List[str]:
        """Get list of processed document IDs"""
        return list(self.processed_documents.keys())
    
    def clear_cache(self) -> None:
        """Clear processed documents cache"""
        self.processed_documents.clear()
        logger.info("Cleared document cache")
    
    async def batch_process_documents(self, document_urls: List[str]) -> Dict[str, Any]:
        """Process multiple documents in batch"""
        try:
            results = {}
            errors = {}
            
            # Process documents concurrently
            tasks = []
            for url in document_urls:
                task = asyncio.create_task(
                    self.process_and_index_document(url),
                    name=f"process_{self._generate_document_id(url)}"
                )
                tasks.append((url, task))
            
            # Wait for all tasks to complete
            for url, task in tasks:
                try:
                    result = await task
                    results[url] = result['document_id']
                except Exception as e:
                    errors[url] = str(e)
                    logger.error(f"Error processing {url}: {str(e)}")
            
            return {
                'successful': results,
                'errors': errors,
                'total_processed': len(results),
                'total_errors': len(errors)
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

