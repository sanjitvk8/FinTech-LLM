from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    TEXT = "text"

class QueryRequest(BaseModel):
    documents: str = Field(..., description="Document URL or content")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer")
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one question is required")
        return v

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    page_number: Optional[int] = None
    section: Optional[str] = None

class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    similarity_score: float
    relevance_score: float

class DecisionResult(BaseModel):
    decision: str = Field(..., description="The final decision (approved/rejected/pending)")
    amount: Optional[float] = Field(None, description="Amount if applicable")
    justification: str = Field(..., description="Detailed explanation of the decision")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision")
    referenced_clauses: List[str] = Field(default_factory=list, description="Referenced document clauses")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to input questions")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class StructuredQuery(BaseModel):
    original_query: str
    extracted_entities: Dict[str, Any]
    query_type: str
    confidence: float
    processed_query: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

