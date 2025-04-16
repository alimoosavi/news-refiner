from typing import Dict, List
import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from config import config
from db.db_manager import DBManager, RawNews
from utils import preprocess_persian_document, NEWS_CATEGORIES

class NewsCategorizer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",
            openai_api_key=config.openai.api_key,
            max_retries=config.openai.max_retries,
            request_timeout=config.openai.timeout
        )
        
        self.response_schema = ResponseSchema(
            name="category",
            description="The category of the news article",
            type="string",
            enum=NEWS_CATEGORIES
        )
        
        self.output_parser = StructuredOutputParser.from_response_schemas([self.response_schema])
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Persian news categorization expert. Categorize the given news text into one of these categories:
            {categories}
            
            Rules:
            1. Consider both the content and context of the news
            2. If multiple categories could apply, choose the most relevant one
            3. Output only the category name exactly as shown in the list
            4. For political news about international relations, use "World News"
            5. For domestic political news, use "Politics"
            
            {format_instructions}
            """),
            ("human", "{text}")
        ])

    async def categorize_batch(self, db_manager: DBManager, batch_size: int = 50) -> Dict[int, str]:
        """
        Fetch and categorize a batch of unprocessed news articles
        
        Args:
            db_manager: Database manager instance
            batch_size: Number of articles to process in one batch
            
        Returns:
            Dictionary mapping news IDs to their categories
        """
        try:
            # Fetch unprocessed news
            news_items = db_manager.get_unprocessed_news(limit=batch_size)
            if not news_items:
                self.logger.info("No unprocessed news found")
                return {}

            self.logger.info(f"Categorizing {len(news_items)} news articles")
            
            # Process and categorize each article
            categorized_news = {}
            
            for item in news_items:
                try:
                    # Preprocess the text
                    processed_text = preprocess_persian_document(item.content)
                    
                    # Format the prompt
                    formatted_prompt = self.prompt.format_messages(
                        categories="\n".join(NEWS_CATEGORIES),
                        format_instructions=self.output_parser.get_format_instructions(),
                        text=processed_text
                    )
                    
                    # Get category from LLM
                    response = await self.llm.apredict_messages(formatted_prompt)
                    parsed_response = self.output_parser.parse(response.content)
                    category = parsed_response['category']
                    
                    categorized_news[item.id] = category
                    
                except Exception as e:
                    self.logger.error(f"Failed to categorize news ID {item.id}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully categorized {len(categorized_news)} articles")
            return categorized_news
            
        except Exception as e:
            self.logger.error(f"Batch categorization failed: {str(e)}")
            raise

    # Add this method to the NewsCategorizer class
    
    async def categorize_text(self, text: str) -> str:
        """
        Categorize a single text query
        """
        try:
            # Use the existing prompt template with NEWS_CATEGORIES
            formatted_prompt = self.prompt.format_messages(
                categories="\n".join(NEWS_CATEGORIES),
                format_instructions=self.output_parser.get_format_instructions(),
                text=text
            )
            
            # Get category from LLM
            response = await self.llm.apredict_messages(formatted_prompt)
            parsed_response = self.output_parser.parse(response.content)
            category = parsed_response['category']
            
            if category not in NEWS_CATEGORIES:
                self.logger.warning(f"Invalid category detected: {category}, defaulting to 'Local News'")
                return "Local News"
                
            return category
            
        except Exception as e:
            self.logger.error(f"Error categorizing text: {str(e)}")
            return "Local News"