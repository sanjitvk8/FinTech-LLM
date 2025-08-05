import logging
import time
from typing import List, Dict, Any
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .llm_service import LLMService
from ..utils.vector_store import VectorStore
from .config import settings

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Main service that orchestrates query processing workflow"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
        # Initialize vector store with a default dimension, will be updated if needed
        default_dimension = self.embedding_service.get_embedding_dimension(method="sentence_transformer")
        self.vector_store = VectorStore(dimension=default_dimension)
        
    async def process_query(self, documents_url: str, questions: List[str]) -> Dict[str, Any]:
        """Main method to process queries against documents"""
        start_time = time.time()
        
        try:
            # Step 1: Process documents
            logger.info("Processing documents...")
            document_data = await self.document_processor.process_document_from_url(documents_url)
            
            # Step 2: Generate embeddings for document chunks
            logger.info("Generating embeddings for document chunks...")
            chunk_texts = [chunk['content'] for chunk in document_data['chunks']]
            
            embedding_method = "auto"
            if not self.embedding_service.has_openai_key:
                embedding_method = "sentence_transformer"
            
            chunk_embeddings = await self.embedding_service.generate_embeddings(chunk_texts, method=embedding_method)

            # Step 3: Build vector store
            logger.info("Building vector store...")
            
            # Dynamically set vector store dimension based on actual embeddings
            if chunk_embeddings:
                actual_dimension = len(chunk_embeddings[0])
                if self.vector_store.dimension != actual_dimension:
                    self.vector_store = VectorStore(dimension=actual_dimension)

            chunk_metadata = []
            for i, chunk in enumerate(document_data['chunks']):
                chunk_metadata.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_url': documents_url,
                    'chunk_index': i,
                    'size': chunk['size']
                })
            
            self.vector_store.add_vectors(chunk_embeddings, document_data['chunks'], chunk_metadata)
            
            # Step 4: Process each question
            answers = []
            for question in questions:
                logger.info(f"Processing question: {question}")
                answer = await self._process_single_question(question)
                answers.append(answer)
            
            processing_time = time.time() - start_time
            
            return {
                'answers': answers,
                'processing_time': processing_time,
                'metadata': {
                    'document_chunks': len(document_data['chunks']),
                    'total_questions': len(questions),
                    'source_url': documents_url,
                    'vector_store_stats': self.vector_store.get_stats()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _process_single_question(self, question: str) -> str:
        """Process a single question against the loaded documents"""
        try:
            # Step 1: Generate embedding for the question
            question_embedding = await self.embedding_service.generate_embeddings([question])
            
            # Step 2: Search for relevant chunks
            search_results = self.vector_store.search(
                question_embedding[0], 
                k=settings.MAX_RETRIEVED_CHUNKS,
                threshold=settings.SIMILARITY_THRESHOLD
            )
            
            # Step 3: Filter and rank results
            relevant_chunks = []
            for result in search_results:
                if result['similarity_score'] >= settings.SIMILARITY_THRESHOLD:
                    relevant_chunks.append(result)
            
            # Step 4: Generate answer using LLM
            if relevant_chunks:
                answer = await self.llm_service.answer_question(question, relevant_chunks)
            else:
                answer = "I couldn't find relevant information in the provided document to answer this question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing single question: {str(e)}")
            return f"An error occurred while processing the question: {str(e)}"
    
    async def process_structured_query(self, query: str, documents_url: str) -> Dict[str, Any]:
        """Process a structured query for decision making (like insurance claims)"""
        start_time = time.time()
        
        try:
            # Step 1: Parse the query
            logger.info("Parsing query...")
            query_info = await self.llm_service.parse_query(query)
            
            # Step 2: Process documents (if not already done)
            logger.info("Processing documents...")
            document_data = await self.document_processor.process_document_from_url(documents_url)
            
            # Step 3: Generate embeddings
            chunk_texts = [chunk['content'] for chunk in document_data['chunks']]
            chunk_embeddings = await self.embedding_service.generate_embeddings(chunk_texts)
            
            # Step 4: Build vector store
            chunk_metadata = []
            for i, chunk in enumerate(document_data['chunks']):
                chunk_metadata.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_url': documents_url,
                    'chunk_index': i,
                    'size': chunk['size']
                })
            
            self.vector_store.add_vectors(chunk_embeddings, document_data['chunks'], chunk_metadata)
            
            # Step 5: Search for relevant information
            processed_query = query_info.get('processed_query', query)
            query_embedding = await self.embedding_service.generate_embeddings([processed_query])
            
            search_results = self.vector_store.search(
                query_embedding[0], 
                k=settings.MAX_RETRIEVED_CHUNKS,
                threshold=0.6  # Lower threshold for structured queries
            )
            
            # Step 6: Make decision
            decision_result = await self.llm_service.make_decision(query_info, search_results)
            
            processing_time = time.time() - start_time
            
            return {
                'query_info': query_info,
                'decision': decision_result,
                'relevant_chunks_count': len(search_results),
                'processing_time': processing_time,
                'metadata': {
                    'document_chunks': len(document_data['chunks']),
                    'similarity_scores': [r['similarity_score'] for r in search_results[:5]]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing structured query: {str(e)}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'vector_store_stats': self.vector_store.get_stats(),
            'embedding_service': {
                'model': settings.EMBEDDING_MODEL,
                'dimension': 1536
            },
            'llm_service': {
                'model': settings.OPENAI_MODEL,
                'max_tokens': settings.MAX_TOKENS_PER_REQUEST
            },
            'configuration': {
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP,
                'similarity_threshold': settings.SIMILARITY_THRESHOLD,
                'max_retrieved_chunks': settings.MAX_RETRIEVED_CHUNKS
            }
        }

