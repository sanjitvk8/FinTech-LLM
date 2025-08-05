import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
import openai
import tiktoken
from .config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with OpenAI GPT models"""
    
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            logger.warning("OpenAI API key not provided. LLM functionality will be limited.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse and structure natural language query"""
        if not self.client:
            logger.error("OpenAI client not initialized - API key missing")
            return {
                "original_query": query,
                "extracted_entities": {},
                "query_type": "general",
                "confidence": 0.0,
                "processed_query": query
            }
        try:
            prompt = f"""
            Parse the following query and extract structured information:
            
            Query: "{query}"
            
            Extract the following information if available:
            - Age (if mentioned)
            - Gender (if mentioned)
            - Medical procedure/condition (if mentioned)
            - Location (if mentioned)
            - Policy duration/age (if mentioned)
            - Insurance type (if mentioned)
            - Any other relevant entities
            
            Also determine the query type (e.g., "coverage_check", "claim_inquiry", "policy_question")
            
            Return the response as a JSON object with the following structure:
            {{
                "extracted_entities": {{
                    "age": "value or null",
                    "gender": "value or null",
                    "procedure": "value or null",
                    "location": "value or null",
                    "policy_duration": "value or null",
                    "insurance_type": "value or null",
                    "other_entities": {{}}
                }},
                "query_type": "category",
                "confidence": 0.0-1.0,
                "processed_query": "reformulated clear query"
            }}
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at parsing insurance and medical queries. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            result["original_query"] = query
            
            return result
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            return {
                "original_query": query,
                "extracted_entities": {},
                "query_type": "general",
                "confidence": 0.5,
                "processed_query": query
            }
    
    async def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Answer a question based on provided context chunks"""
        if not self.client:
            logger.error("OpenAI client not initialized - API key missing")
            return "I apologize, but I cannot answer questions without a valid OpenAI API key. Please configure the API key and try again."
        try:
            # Prepare context from chunks
            context_parts = []
            for i, chunk in enumerate(context_chunks[:5]):  # Limit to top 5 chunks
                chunk_content = chunk.get('chunk', {}).get('content', '')
                if chunk_content:
                    context_parts.append(f"[Context {i+1}]\n{chunk_content}")
            
            context = "\n\n".join(context_parts)
            
            # Truncate context if it's too long
            context_tokens = len(self.encoding.encode(context))
            if context_tokens > 3000:
                # Truncate context to fit within token limits
                truncated_context = self.encoding.decode(
                    self.encoding.encode(context)[:3000]
                )
                context = truncated_context
            
            prompt = f"""
            Based on the following context from policy documents, answer the question accurately and concisely.
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Answer based only on the information provided in the context
            2. If the information is not available in the context, state that clearly
            3. Be specific and include relevant details like time periods, amounts, conditions
            4. Keep the answer concise but complete
            5. Do not make assumptions beyond what's stated in the context
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate, concise answers based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    async def make_decision(self, query_info: Dict[str, Any], 
                          relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision based on query and relevant policy chunks"""
        try:
            # Prepare context
            context_parts = []
            clause_references = []
            
            for i, chunk in enumerate(relevant_chunks[:10]):
                chunk_content = chunk.get('chunk', {}).get('content', '')
                if chunk_content:
                    context_parts.append(f"[Clause {i+1}]\n{chunk_content}")
                    clause_references.append(f"Clause {i+1}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""
            You are an insurance claim processor. Based on the query information and policy clauses below, make a decision.
            
            Query Information:
            - Original Query: {query_info.get('original_query', '')}
            - Extracted Entities: {json.dumps(query_info.get('extracted_entities', {}), indent=2)}
            - Query Type: {query_info.get('query_type', '')}
            
            Policy Context:
            {context}
            
            Based on this information, provide a decision in the following JSON format:
            {{
                "decision": "approved|rejected|pending|covered|not_covered",
                "amount": null or numeric_value,
                "justification": "detailed explanation of the decision with specific references to policy clauses",
                "confidence_score": 0.0-1.0,
                "referenced_clauses": ["list of clause numbers that support this decision"]
            }}
            
            Guidelines:
            1. Be specific about which clauses support your decision
            2. If information is insufficient, mark as "pending" and explain what's needed
            3. Include specific amounts, waiting periods, or conditions mentioned in the policy
            4. Provide clear reasoning for the decision
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert insurance claim processor. Always respond with valid JSON that follows the specified format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate the result structure
            required_fields = ["decision", "justification", "confidence_score"]
            for field in required_fields:
                if field not in result:
                    result[field] = "unknown" if field == "decision" else "No information available" if field == "justification" else 0.5
            
            if "referenced_clauses" not in result:
                result["referenced_clauses"] = clause_references[:3]  # Default to first 3 clauses
            
            return result
        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            return {
                "decision": "error",
                "amount": None,
                "justification": f"An error occurred while processing the decision: {str(e)}",
                "confidence_score": 0.0,
                "referenced_clauses": []
            }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text.split()) * 1.3  # Rough approximation
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text"""
        try:
            if len(text) < max_length:
                return text
            
            prompt = f"""
            Summarize the following text in {max_length} characters or less, maintaining the key information:
            
            {text}
            """
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",  # Use cheaper model for summaries
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return text[:max_length]

