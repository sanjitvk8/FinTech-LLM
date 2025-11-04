import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from ..core.llm_service import LLMService
from ..core.config import settings
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class DecisionType(str, Enum):
    COVERAGE_CHECK = "coverage_check"
    CLAIM_APPROVAL = "claim_approval"
    POLICY_INQUIRY = "policy_inquiry"
    ELIGIBILITY_CHECK = "eligibility_check"
    PREMIUM_CALCULATION = "premium_calculation"

class DecisionService:
    """Service for making intelligent decisions based on document content and queries"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.decision_cache = {}
        
    async def make_coverage_decision(self, query_info: Dict[str, Any], 
                                   relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a coverage decision for insurance queries"""
        try:
            # Extract key information from query
            entities = query_info.get('extracted_entities', {})
            procedure = entities.get('procedure', '')
            age = entities.get('age', '')
            policy_duration = entities.get('policy_duration', '')
            location = entities.get('location', '')
            
            # Prepare context for decision making
            context_chunks = self._prepare_context_chunks(relevant_chunks, max_chunks=8)
            
            # Check for specific coverage rules
            coverage_rules = await self._extract_coverage_rules(context_chunks, procedure)
            
            # Check for exclusions
            exclusions = await self._check_exclusions(context_chunks, entities)
            
            # Check waiting periods
            waiting_period_check = await self._check_waiting_periods(
                context_chunks, entities, policy_duration
            )
            
            # Make final decision
            decision_result = await self._make_final_coverage_decision(
                query_info, coverage_rules, exclusions, waiting_period_check, context_chunks
            )
            
            # Add decision metadata
            decision_result['decision_type'] = DecisionType.COVERAGE_CHECK
            decision_result['processed_entities'] = entities
            decision_result['context_chunks_used'] = len(context_chunks)
            
            return decision_result
            
        except Exception as e:
            logger.error(f"Error making coverage decision: {str(e)}")
            return self._create_error_decision(str(e))
    
    async def make_claim_decision(self, claim_info: Dict[str, Any], 
                                policy_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a claim approval/rejection decision"""
        try:
            # Extract claim details
            claim_amount = claim_info.get('claim_amount', 0)
            claim_type = claim_info.get('claim_type', '')
            supporting_documents = claim_info.get('documents', [])
            
            # Prepare context
            context_chunks = self._prepare_context_chunks(policy_chunks, max_chunks=10)
            
            # Check policy limits
            policy_limits = await self._check_policy_limits(context_chunks, claim_amount, claim_type)
            
            # Verify claim eligibility
            eligibility_check = await self._verify_claim_eligibility(context_chunks, claim_info)
            
            # Check for fraud indicators
            fraud_check = await self._check_fraud_indicators(claim_info, context_chunks)
            
            # Calculate payout amount
            payout_calculation = await self._calculate_payout(
                claim_info, policy_limits, context_chunks
            )
            
            # Make final decision
            decision_result = await self._make_final_claim_decision(
                claim_info, policy_limits, eligibility_check, fraud_check, 
                payout_calculation, context_chunks
            )
            
            decision_result['decision_type'] = DecisionType.CLAIM_APPROVAL
            decision_result['claim_analysis'] = {
                'policy_limits': policy_limits,
                'eligibility': eligibility_check,
                'fraud_score': fraud_check.get('risk_score', 0.0),
                'calculated_payout': payout_calculation
            }
            
            return decision_result
            
        except Exception as e:
            logger.error(f"Error making claim decision: {str(e)}")
            return self._create_error_decision(str(e))
    
    async def _extract_coverage_rules(self, context_chunks: List[Dict[str, Any]], 
                                    procedure: str) -> Dict[str, Any]:
        """Extract specific coverage rules for a procedure"""
        try:
            context = self._format_context_for_llm(context_chunks)
            
            prompt = f"""
            Based on the following policy context, extract specific coverage rules for "{procedure}":
            
            {context}
            
            Please identify:
            1. Is this procedure covered? (yes/no/conditional)
            2. What are the specific conditions or requirements?
            3. Are there any coverage limits or caps?
            4. What waiting periods apply?
            5. Any co-payment or deductible requirements?
            
            Return as JSON:
            {{
                "is_covered": "yes|no|conditional",
                "conditions": ["list of conditions"],
                "coverage_limit": "amount or 'unlimited'",
                "waiting_period": "duration or 'none'",
                "copayment": "amount or percentage",
                "deductible": "amount",
                "additional_notes": "any other relevant information"
            }}
            """
            
            if self.llm_service.provider == 'groq':
                response = await asyncio.to_thread(
                    self.llm_service.groq_client.chat.completions.create,
                    model=self.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert insurance policy analyst. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.llm_service.openai_client.chat.completions.create,
                    model=self.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert insurance policy analyst. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
                )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error extracting coverage rules: {str(e)}")
            return {
                "is_covered": "unknown",
                "conditions": [],
                "coverage_limit": "unknown",
                "waiting_period": "unknown",
                "copayment": "unknown",
                "deductible": "unknown",
                "additional_notes": f"Error analyzing coverage: {str(e)}"
            }
    
    async def _check_exclusions(self, context_chunks: List[Dict[str, Any]], 
                              entities: Dict[str, Any]) -> Dict[str, Any]:
        """Check for policy exclusions that might apply"""
        try:
            context = self._format_context_for_llm(context_chunks)
            
            prompt = f"""
            Based on the policy context and the following details, identify any exclusions that might apply:
            
            Policy Context:
            {context}
            
            Case Details:
            - Age: {entities.get('age', 'Not specified')}
            - Procedure: {entities.get('procedure', 'Not specified')}
            - Location: {entities.get('location', 'Not specified')}
            - Policy Duration: {entities.get('policy_duration', 'Not specified')}
            
            Identify:
            1. Any exclusions that apply to this case
            2. The specific exclusion clauses
            3. Whether the exclusion is absolute or conditional
            
            Return as JSON:
            {{
                "applicable_exclusions": [
                    {{
                        "exclusion_type": "type of exclusion",
                        "description": "description",
                        "clause_reference": "reference to policy clause",
                        "is_absolute": true/false
                    }}
                ],
                "exclusion_applies": true/false,
                "exclusion_reason": "explanation if exclusions apply"
            }}
            """
            
            if self.llm_service.provider == 'groq':
                response = await asyncio.to_thread(
                    self.llm_service.groq_client.chat.completions.create,
                    model=self.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying insurance policy exclusions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.llm_service.openai_client.chat.completions.create,
                    model=self.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying insurance policy exclusions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600
                )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error checking exclusions: {str(e)}")
            return {
                "applicable_exclusions": [],
                "exclusion_applies": False,
                "exclusion_reason": f"Error checking exclusions: {str(e)}"
            }
    
    async def _check_waiting_periods(self, context_chunks: List[Dict[str, Any]], 
                                   entities: Dict[str, Any], 
                                   policy_duration: str) -> Dict[str, Any]:
        """Check if waiting periods are satisfied"""
        try:
            context = self._format_context_for_llm(context_chunks)
            
            prompt = f"""
            Analyze waiting periods for this case:
            
            Policy Context:
            {context}
            
            Case Details:
            - Procedure: {entities.get('procedure', 'Not specified')}
            - Policy Duration: {policy_duration}
            - Age: {entities.get('age', 'Not specified')}
            
            Determine:
            1. What waiting periods apply to this procedure?
            2. Has the waiting period been satisfied?
            3. How much longer until waiting period is satisfied (if not met)?
            
            Return as JSON:
            {{
                "waiting_periods": [
                    {{
                        "type": "general|specific procedure|pre-existing",
                        "duration": "duration required",
                        "description": "what this waiting period covers"
                    }}
                ],
                "waiting_period_satisfied": true/false,
                "remaining_wait_time": "time remaining or 'none'",
                "analysis": "detailed explanation"
            }}
            """
            
            if self.llm_service.provider == 'groq':
                response = await asyncio.to_thread(
                    self.llm_service.groq_client.chat.completions.create,
                    model=self.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing insurance waiting periods."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
            else:  # openai
                response = await asyncio.to_thread(
                    self.llm_service.openai_client.chat.completions.create,
                    model=self.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing insurance waiting periods."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600
                )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error checking waiting periods: {str(e)}")
            return {
                "waiting_periods": [],
                "waiting_period_satisfied": True,
                "remaining_wait_time": "unknown",
                "analysis": f"Error analyzing waiting periods: {str(e)}"
            }
    
    async def _make_final_coverage_decision(self, query_info: Dict[str, Any],
                                          coverage_rules: Dict[str, Any],
                                          exclusions: Dict[str, Any],
                                          waiting_periods: Dict[str, Any],
                                          context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make the final coverage decision based on all analysis"""
        try:
            # Determine decision based on analysis
            decision = "approved"
            justification_parts = []
            confidence = 0.8
            
            # Check coverage
            if coverage_rules.get('is_covered') == 'no':
                decision = "rejected"
                justification_parts.append("The requested procedure is not covered under this policy.")
                confidence = 0.9
            elif coverage_rules.get('is_covered') == 'conditional':
                decision = "conditional"
                conditions = coverage_rules.get('conditions', [])
                justification_parts.append(f"Coverage is conditional on: {', '.join(conditions)}")
                confidence = 0.7
            
            # Check exclusions
            if exclusions.get('exclusion_applies', False):
                decision = "rejected"
                justification_parts.append(f"Excluded under policy: {exclusions.get('exclusion_reason', '')}")
                confidence = 0.9
            
            # Check waiting periods
            if not waiting_periods.get('waiting_period_satisfied', True):
                decision = "rejected"
                remaining_time = waiting_periods.get('remaining_wait_time', 'unknown')
                justification_parts.append(f"Waiting period not satisfied. Remaining time: {remaining_time}")
                confidence = 0.95
            
            # Compile justification
            if not justification_parts:
                justification_parts.append("All coverage requirements are met.")
            
            # Get referenced clauses
            referenced_clauses = self._extract_clause_references(context_chunks)
            
            return {
                "decision": decision,
                "amount": None,  # Coverage decisions don't typically have amounts
                "justification": " ".join(justification_parts),
                "confidence_score": confidence,
                "referenced_clauses": referenced_clauses[:5],  # Limit to top 5
                "detailed_analysis": {
                    "coverage_rules": coverage_rules,
                    "exclusions": exclusions,
                    "waiting_periods": waiting_periods
                }
            }
            
        except Exception as e:
            logger.error(f"Error making final coverage decision: {str(e)}")
            return self._create_error_decision(str(e))
    
    def _prepare_context_chunks(self, chunks: List[Dict[str, Any]], 
                              max_chunks: int = 10) -> List[Dict[str, Any]]:
        """Prepare and limit context chunks for LLM processing"""
        try:
            # Sort by similarity score if available
            sorted_chunks = sorted(
                chunks, 
                key=lambda x: x.get('similarity_score', 0.0), 
                reverse=True
            )
            
            return sorted_chunks[:max_chunks]
            
        except Exception as e:
            logger.error(f"Error preparing context chunks: {str(e)}")
            return chunks[:max_chunks]
    
    def _format_context_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string for LLM"""
        try:
            context_parts = []
            
            for i, chunk in enumerate(chunks):
                chunk_content = chunk.get('chunk', {}).get('content', '')
                if chunk_content:
                    context_parts.append(f"[Context {i+1}]\n{chunk_content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return "Error formatting context for analysis."
    
    def _extract_clause_references(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract clause references from context chunks"""
        try:
            references = []
            
            for i, chunk in enumerate(chunks):
                # Try to extract clause identifiers from metadata or content
                metadata = chunk.get('metadata', {})
                chunk_id = metadata.get('chunk_id', f"Clause {i+1}")
                references.append(chunk_id)
            
            return references
            
        except Exception as e:
            logger.error(f"Error extracting clause references: {str(e)}")
            return [f"Clause {i+1}" for i in range(min(len(chunks), 5))]
    
    def _create_error_decision(self, error_message: str) -> Dict[str, Any]:
        """Create an error decision response"""
        return {
            "decision": "error",
            "amount": None,
            "justification": f"An error occurred during decision processing: {error_message}",
            "confidence_score": 0.0,
            "referenced_clauses": [],
            "error": error_message
        }
    
    # Additional methods for claim processing (simplified versions)
    async def _check_policy_limits(self, context_chunks: List[Dict[str, Any]], 
                                 claim_amount: float, claim_type: str) -> Dict[str, Any]:
        """Check policy limits for claim processing"""
        # Implementation would analyze policy limits
        return {"within_limits": True, "limit_amount": "unlimited", "remaining_coverage": "unlimited"}
    
    async def _verify_claim_eligibility(self, context_chunks: List[Dict[str, Any]], 
                                      claim_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify claim eligibility"""
        # Implementation would verify eligibility criteria
        return {"eligible": True, "eligibility_reasons": ["All criteria met"]}
    
    async def _check_fraud_indicators(self, claim_info: Dict[str, Any], 
                                    context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for potential fraud indicators"""
        # Implementation would analyze fraud risk
        return {"risk_score": 0.1, "risk_factors": [], "requires_investigation": False}
    
    async def _calculate_payout(self, claim_info: Dict[str, Any], 
                              policy_limits: Dict[str, Any], 
                              context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate claim payout amount"""
        # Implementation would calculate payout based on policy terms
        claim_amount = claim_info.get('claim_amount', 0)
        return {"payout_amount": claim_amount, "deductions": 0, "net_payout": claim_amount}
    
    async def _make_final_claim_decision(self, claim_info: Dict[str, Any], 
                                       policy_limits: Dict[str, Any],
                                       eligibility: Dict[str, Any], 
                                       fraud_check: Dict[str, Any],
                                       payout_calc: Dict[str, Any], 
                                       context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make final claim decision"""
        # Simplified implementation
        decision = "approved" if eligibility.get('eligible', False) else "rejected"
        amount = payout_calc.get('net_payout', 0) if decision == "approved" else 0
        
        return {
            "decision": decision,
            "amount": amount,
            "justification": "Claim meets all policy requirements." if decision == "approved" else "Claim does not meet eligibility criteria.",
            "confidence_score": 0.8,
            "referenced_clauses": self._extract_clause_references(context_chunks)
        }