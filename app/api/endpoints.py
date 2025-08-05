import logging
import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..core.query_processor import QueryProcessor
from .models import QueryRequest, QueryResponse, ErrorResponse
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# Initialize services
query_processor = QueryProcessor()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the authorization token"""
    expected_token = "d5f4ea72e4cce0351417c18d19b196b40e7dbe8f29d2ece3c9b01f6680160650"
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    return credentials.credentials

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document queries
    """
    try:
        start_time = time.time()
        
        logger.info(f"Processing query with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Process the query
        result = await query_processor.process_query(
            documents_url=request.documents,
            questions=request.questions
        )
        
        processing_time = time.time() - start_time
        
        # Log successful processing
        logger.info(f"Successfully processed query in {processing_time:.2f} seconds")
        
        # Add background task for cleanup if needed
        background_tasks.add_task(cleanup_resources)
        
        return QueryResponse(
            answers=result['answers'],
            processing_time=processing_time,
            metadata=result.get('metadata', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/process-structured-query")
async def process_structured_query(
    query: str,
    documents_url: str,
    token: str = Depends(verify_token)
):
    """
    Endpoint for processing structured queries (like insurance claims)
    """
    try:
        logger.info(f"Processing structured query: {query}")
        
        result = await query_processor.process_structured_query(query, documents_url)
        
        return {
            "status": "success",
            "query_info": result['query_info'],
            "decision": result['decision'],
            "processing_time": result['processing_time'],
            "metadata": result.get('metadata', {})
        }
        
    except Exception as e:
        logger.error(f"Error processing structured query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "system_stats": query_processor.get_system_stats()
    }

@router.get("/stats")
async def get_system_stats(token: str = Depends(verify_token)):
    """Get system statistics"""
    try:
        stats = query_processor.get_system_stats()
        return {
            "status": "success",
            "stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def cleanup_resources():
    """Background task for resource cleanup"""
    try:
        # Clear vector store to free memory
        query_processor.vector_store.clear()
        logger.info("Cleaned up resources")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {str(e)}")

# Error handlers are now handled by the main app in main.py

