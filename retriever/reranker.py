from typing import List, Dict, Any
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
    relevance_explanation: str
    metadata: Dict[str, Any]

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
        
        self.parser = StructuredOutputParser.from_response_schemas([self.reranking_schema])
        
        self.reranking_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian news search expert. Rerank the given search results based on their relevance to the query.
            
            Follow these rules:
            1. Analyze semantic relevance between query and content
            2. Consider recency and source credibility
            3. Evaluate content quality and completeness
            4. Score each result from 0.0 to 1.0
            5. Provide a brief explanation for each ranking
            
            For each result, return:
            - score: float between 0.0 and 1.0
            - relevance_explanation: brief explanation in Persian
            
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
            
            # Create ranked results
            ranked_results = []
            for rank_info, original_result in zip(parsed["reranked_results"], results):
                ranked_results.append(
                    RankedResult(
                        content=original_result["content"],
                        score=float(rank_info["score"]),
                        relevance_explanation=rank_info["relevance_explanation"],
                        metadata=original_result["metadata"]
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