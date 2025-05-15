from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from config import config

logger = logging.getLogger(__name__)

@dataclass
class RankedResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    relevance_explanation: Optional[str] = None

class Reranker:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=config.reranker.temperature,
            model=config.reranker.model,
            openai_api_key=config.openai.api_key
        )
        
        # Schema for reranking output
        self.reranking_schema = ResponseSchema(
            name="reranked_results",
            description="List of reranked results with scores and explanations",
            type="list[dict]"
        )
        
        # Initialize the parser with the schema
        # Change from direct list to from_response_schemas method
        self.parser = StructuredOutputParser.from_response_schemas([self.reranking_schema])
        
        self.reranking_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian news search expert. Your task is to carefully evaluate and rerank search results based on their relevance to the user's query.
            
            FILTERING CRITERIA - First, strictly evaluate each result and filter out irrelevant content:
            - Score 0.0 for results that are not directly related to the main topic of the query
            - Score 0.0 for results that only superficially mention query terms without substantive information
            - Score 0.0 for outdated information that has been superseded by more recent developments
            - Score 0.0 for content that is misleading or contains factual inaccuracies related to the query
            - Score 0.0 for content that is too general when the query asks for specific information
            
            RANKING CRITERIA - For relevant results, assign scores based on these factors:
            - Score 0.9-1.0: Perfect match that comprehensively addresses the query with authoritative information
            - Score 0.7-0.8: Strong match with significant relevant details that answers most aspects of the query
            - Score 0.4-0.6: Moderate match that addresses some aspects of the query with reasonable depth
            - Score 0.1-0.3: Minimal relevance that touches on the query topic but lacks depth or completeness
            
            EVALUATION FACTORS - Consider these dimensions when scoring:
            1. Topical Relevance (40%):
               - How directly does the content address the specific query topic?
               - Does it cover the exact subject matter or only tangentially related topics?
            
            2. Information Quality (30%):
               - How comprehensive, accurate and detailed is the information?
               - Does it provide unique insights or just basic information?
               - Is the information from a credible source?
            
            3. Temporal Relevance (20%):
               - How recent and timely is the information relative to the query needs?
               - For current events, newer content should generally score higher
               - For historical topics, authoritative content may score higher regardless of age
            
            4. Content Utility (10%):
               - How useful would this information be to someone asking this specific query?
               - Does it provide actionable insights or answer likely follow-up questions?
            
            For each result, you MUST return:
            - score: float between 0.0 and 1.0 (0.0 means irrelevant and will be filtered out)
            - explanation: detailed explanation of why this result received its score, referencing specific content elements
            
            {format_instructions}
            """),
            ("human", """Query: {query}

            Search Results:
            {results}""")
        ])

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[RankedResult]:
        """Rerank search results using LLM"""
        try:
            if not results:
                return []
            
            # Format results for the prompt
            results_text = "\n\n".join([
                f"Content {i+1}:\n{result['content'][:500]}..."
                for i, result in enumerate(results)
            ])
            
            # Generate reranking prompt
            formatted_prompt = self.reranking_prompt.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                query=query,
                results=results_text
            )
            
            # Get LLM response
            response = await self.llm.apredict_messages(formatted_prompt)
            parsed = self.parser.parse(response.content)
            
            # Create ranked results and filter out irrelevant ones (score = 0)
            ranked_results = []
            for rank_info, original_result in zip(parsed["reranked_results"], results):
                score = float(rank_info["score"])
                if score > 0:  # Only include relevant results
                    ranked_results.append(
                        RankedResult(
                            content=original_result["content"],
                            score=score,
                            metadata=original_result["metadata"],
                            relevance_explanation=rank_info.get("explanation", "")  # Add explanation from reranking
                        )
                    )
            
            # Sort by score and limit results
            ranked_results.sort(key=lambda x: x.score, reverse=True)
            if top_k:
                ranked_results = ranked_results[:top_k]
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            return results  # Return original results on error