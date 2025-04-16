from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from config import config
from utils import preprocess_persian_document


@dataclass
class NewsChunk:
    content: str
    keywords: List[str]
    is_meaningful: bool = True
    website_link: Optional[str] = None


@dataclass
class QueryChunk:
    content: str
    keywords: List[str]


class NewsPreprocessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",
            openai_api_key=config.openai.api_key,
            max_retries=config.openai.max_retries,
            request_timeout=config.openai.timeout
        )

        # Schema for content cleaning and validation
        self.cleaning_schema = ResponseSchema(
            name="cleaned_text",
            description="The cleaned and validated news text",
            type="string"
        )

        self.validation_schema = ResponseSchema(
            name="is_valid",
            description="Whether the text is a valid news article",
            type="boolean"
        )

        self.website_link_schema = ResponseSchema(
            name="website_link",
            description="The main website link from the text, if any",
            type="string"
        )

        # Initialize parsers
        self.cleaning_parser = StructuredOutputParser.from_response_schemas(
            [self.cleaning_schema, self.validation_schema, self.website_link_schema]
        )

        # Update the cleaning prompt to handle website links
        self.cleaning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian news text cleaner and validator. Clean and validate the given text following these rules:

            Cleaning Rules:
            1. Extract the main website URL if present
            2. Remove redundant whitespace and formatting
            3. Fix obvious Persian text errors
            4. Remove any advertising content
            5. Preserve important quotes and numbers
            6. Keep only the main news content
            
            Validation Rules:
            1. Text should contain actual news content
            2. Reject promotional or advertising content
            3. Reject content that's just a list of links
            4. Reject content that's too short or meaningless
            5. Reject duplicate content within the text
            
            {format_instructions}
            """),
            ("human", "{text}")
        ])

        # Schema for keywords
        self.keywords_schema = ResponseSchema(
            name="keywords",
            description="List of relevant keywords",
            type="list[str]"
        )

        # Initialize parsers
        self.cleaning_parser = StructuredOutputParser.from_response_schemas(
            [self.cleaning_schema, self.validation_schema]
        )
        self.keywords_parser = StructuredOutputParser.from_response_schemas(
            [self.keywords_schema]
        )

        # Prompt templates
        self.cleaning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian news text cleaner and validator. Clean and validate the given text following these rules:

            Cleaning Rules:
            1. Remove any website URLs while preserving their context
            2. Remove redundant whitespace and formatting
            3. Fix obvious Persian text errors
            4. Remove any advertising content
            5. Preserve important quotes and numbers
            6. Keep only the main news content
            
            Validation Rules:
            1. Text should contain actual news content
            2. Reject promotional or advertising content
            3. Reject content that's just a list of links
            4. Reject content that's too short or meaningless
            5. Reject duplicate content within the text
            
            {format_instructions}
            """),
            ("human", "{text}")
        ])

        self.keywords_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian text analysis expert. Generate relevant keywords for the given text following these rules:

            1. Extract 3-7 most important keywords
            2. Include both specific terms and general topics
            3. Keywords should be in Persian
            4. Include named entities (people, organizations, locations)
            5. Keywords should help in categorizing and searching the content
            
            {format_instructions}
            """),
            ("human", "{text}")
        ])

    async def _clean_and_validate_text(self, text: str) -> Tuple[str, bool, Optional[str]]:
        """Clean and validate the news text"""
        try:
            formatted_prompt = self.cleaning_prompt.format_messages(
                format_instructions=self.cleaning_parser.get_format_instructions(),
                text=text
            )

            response = await self.llm.apredict_messages(formatted_prompt)
            parsed = self.cleaning_parser.parse(response.content)

            return parsed["cleaned_text"], parsed["is_valid"], parsed.get("website_link")

        except Exception as e:
            self.logger.error(f"Text cleaning failed: {str(e)}")
            return text, False, None

    async def process_news(self, content: str) -> List[NewsChunk]:
        try:
            processed_text = preprocess_persian_document(content)

            # Clean and validate
            cleaned_text, is_valid, website_link = await self._clean_and_validate_text(processed_text)
            if not is_valid:
                self.logger.info("News content failed validation")
                return []

            # Split into chunks
            chunks = self._split_into_chunks(cleaned_text)
            processed_chunks = []

            # Process each chunk
            for chunk in chunks:
                keywords = await self._generate_keywords(chunk)

                processed_chunks.append(NewsChunk(
                    content=chunk,
                    keywords=keywords,
                    website_link=website_link  # Add website link to each chunk
                ))

            return processed_chunks

        except Exception as e:
            self.logger.error(f"News processing failed: {str(e)}")
            return []

    async def _generate_keywords(self, text: str) -> List[str]:
        """Generate keywords for a text chunk"""
        try:
            formatted_prompt = self.keywords_prompt.format_messages(
                format_instructions=self.keywords_parser.get_format_instructions(),
                text=text
            )

            response = await self.llm.apredict_messages(formatted_prompt)
            parsed = self.keywords_parser.parse(response.content)

            return parsed["keywords"]

        except Exception as e:
            self.logger.error(f"Keyword generation failed: {str(e)}")
            return []

    def _split_into_chunks(self, text: str, max_tokens: int = 1000) -> List[str]:
        """Split text into semantic chunks"""
        # For short texts, return as single chunk
        if len(text.split()) < max_tokens // 4:
            return [text]

        # Split on paragraph breaks first
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para.split())

            if current_length + para_length <= max_tokens:
                current_chunk.append(para)
                current_length += para_length
            else:
                # Join current chunk and add to chunks
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks


class QueryPreprocessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",
            openai_api_key=config.openai.api_key,
            max_retries=config.openai.max_retries,
            request_timeout=config.openai.timeout
        )

        # Schema for keywords
        self.keywords_schema = ResponseSchema(
            name="keywords",
            description="List of relevant keywords",
            type="list[str]"
        )

        self.keywords_parser = StructuredOutputParser.from_response_schemas(
            [self.keywords_schema]
        )

        self.keywords_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian text analysis expert. Generate relevant keywords for the given query following these rules:

            1. Extract 3-5 most important keywords
            2. Include both specific terms and general topics
            3. Keywords should be in Persian
            4. Include named entities (people, organizations, locations)
            5. Keywords should help in finding relevant content
            
            {format_instructions}
            """),
            ("human", "{text}")
        ])

    async def process_query(self, query: str) -> Optional[QueryChunk]:
        """Process a search query to extract keywords"""
        try:
            # Clean and preprocess the query
            processed_query = preprocess_persian_document(query)

            # Generate keywords
            keywords = await self._generate_keywords(processed_query)

            if not keywords:
                self.logger.warning("No keywords generated for query")
                return None

            return QueryChunk(
                content=processed_query,
                keywords=keywords
            )

        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            return None

    async def _generate_keywords(self, text: str) -> List[str]:
        """Generate keywords for the query"""
        try:
            formatted_prompt = self.keywords_prompt.format_messages(
                format_instructions=self.keywords_parser.get_format_instructions(),
                text=text
            )

            response = await self.llm.apredict_messages(formatted_prompt)
            parsed = self.keywords_parser.parse(response.content)

            return parsed["keywords"]

        except Exception as e:
            self.logger.error(f"Keyword generation failed: {str(e)}")
            return []
