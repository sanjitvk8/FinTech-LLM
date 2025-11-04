# app/core/llm_service.py

import logging
import json
import asyncio
from typing import List, Dict, Any

# --- NEW/UPDATED IMPORTS ---
from groq import Groq  # ✅ Correct official Groq SDK import
import openai
import tiktoken
from .config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with both Groq and OpenAI models"""

    def __init__(self):
        # Initialize both clients if API keys are available
        self.groq_client = None
        self.openai_client = None
        
        # Check which provider to use
        provider = settings.LLM_PROVIDER.lower()
        
        if provider in ['groq', 'auto'] and settings.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                self.model = settings.GROQ_MODEL
                self.provider = 'groq'
                logger.info("Using Groq as LLM provider")
            except Exception as e:
                logger.warning(f"Groq client initialization failed: {str(e)}")
                if provider == 'groq':
                    logger.error("Groq provider specified but failed to initialize. LLM functionality will be limited.")
        
        if (provider in ['openai', 'auto'] and settings.OPENAI_API_KEY and self.groq_client is None):
            try:
                self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                self.model = settings.OPENAI_MODEL
                self.provider = 'openai'
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.info("Using OpenAI as LLM provider")
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {str(e)}")
                if provider == 'openai':
                    logger.error("OpenAI provider specified but failed to initialize. LLM functionality will be limited.")
        
        # If no clients are available, set up a fallback method
        if self.groq_client is None and self.openai_client is None:
            logger.warning("No LLM providers available. LLM functionality will be limited.")
            self.provider = 'none'

    # Helper: Rough token count (for Groq since it doesn't have tiktoken support for all models)
    def _approximate_token_count(self, text: str) -> int:
        """Approximate token count (roughly words * 1.3)."""
        return int(len(text.split()) * 1.3)

    # -------------------------------------------------------
    # QUERY PARSING
    # -------------------------------------------------------
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse and structure natural language query"""
        if self.provider == 'none':
            logger.error("No LLM provider available")
            return {
                "original_query": query,
                "extracted_entities": {},
                "query_type": "general",
                "confidence": 0.0,
                "processed_query": query,
            }

        try:
            prompt = f"""
            Parse the following query and extract structured information:

            Query: "{query}"

            Extract the following information if available:
            - Age
            - Gender
            - Medical procedure/condition
            - Location
            - Policy duration
            - Insurance type
            - Other relevant entities

            Determine the query type (e.g., coverage_check, claim_inquiry, policy_question)

            Return as JSON:
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

            if self.provider == 'groq':
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert at parsing insurance and medical queries. "
                                "Always respond with valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
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
                "processed_query": query,
            }

    # -------------------------------------------------------
    # QUESTION ANSWERING
    # -------------------------------------------------------
    async def answer_question(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Answer a question based on provided context chunks"""
        if self.provider == 'none':
            logger.error("No LLM provider available")
            return (
                "I cannot answer questions without a valid API key. "
                "Please configure the API key and try again."
            )

        try:
            context_parts = []
            for i, chunk in enumerate(context_chunks[:5]):
                chunk_content = chunk.get("chunk", {}).get("content", "")
                if chunk_content:
                    context_parts.append(f"[Context {i+1}]\n{chunk_content}")

            context = "\n\n".join(context_parts)

            # Truncate context if too long
            if self.provider == 'openai':
                context_tokens = len(self.encoding.encode(context))
                if context_tokens > 3000:
                    # Truncate context to fit within token limits
                    truncated_context = self.encoding.decode(
                        self.encoding.encode(context)[:3000]
                    )
                    context = truncated_context
            else:  # groq
                context_tokens = self._approximate_token_count(context)
                if context_tokens > 3000:
                    context = " ".join(context.split()[:int(3000 / 1.3)])

            prompt = f"""
            Based on the following policy context, answer accurately and concisely.

            Context:
            {context}

            Question: {question}

            Guidelines:
            1. Answer only from the provided context.
            2. If info is missing, clearly say so.
            3. Be specific, concise, and factual.
            """

            if self.provider == 'groq':
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert insurance policy analyst. Be factual and concise.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
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
            return f"I encountered an error while processing your question: {str(e)}"

    # -------------------------------------------------------
    # DECISION MAKING
    # -------------------------------------------------------
    async def make_decision(
        self, query_info: Dict[str, Any], relevant_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make a decision based on query and relevant policy chunks"""
        if self.provider == 'none':
            logger.error("No LLM provider available")
            return {"decision": "error", "justification": "LLM client not initialized."}

        try:
            context_parts = []
            clause_references = []

            for i, chunk in enumerate(relevant_chunks[:10]):
                chunk_content = chunk.get("chunk", {}).get("content", "")
                if chunk_content:
                    context_parts.append(f"[Clause {i+1}]\n{chunk_content}")
                    clause_references.append(f"Clause {i+1}")

            context = "\n\n".join(context_parts)

            prompt = f"""
            You are an insurance claim processor. Based on the query and clauses, make a decision.

            Query Information:
            {json.dumps(query_info, indent=2)}

            Policy Context:
            {context}

            Respond in JSON:
            {{
                "decision": "approved|rejected|pending|covered|not_covered",
                "amount": null or numeric,
                "justification": "detailed explanation",
                "confidence_score": 0.0-1.0,
                "referenced_clauses": ["list of clause numbers"]
            }}
            """

            if self.provider == 'groq':
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert insurance claim processor. "
                                "Always respond with valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert insurance claim processor. Always respond with valid JSON that follows the specified format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )

            result = json.loads(response.choices[0].message.content)

            required_fields = ["decision", "justification", "confidence_score"]
            for field in required_fields:
                if field not in result:
                    if field == "decision":
                        result[field] = "unknown"
                    elif field == "justification":
                        result[field] = "No information available"
                    else:
                        result[field] = 0.5

            if "referenced_clauses" not in result:
                result["referenced_clauses"] = clause_references[:3]

            return result

        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            return {
                "decision": "error",
                "amount": None,
                "justification": f"Error while processing: {str(e)}",
                "confidence_score": 0.0,
                "referenced_clauses": [],
            }

    # -------------------------------------------------------
    # SUMMARIZATION
    # -------------------------------------------------------
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a concise summary of given text"""
        if self.provider == 'none':
            return text[:max_length]

        try:
            if len(text) < max_length:
                return text

            prompt = f"""
            Summarize the following text concisely, keeping all key details:

            {text}
            """

            if self.provider == 'groq':
                response = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert summarizer. Limit output to around 100 tokens.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=100,
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
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

