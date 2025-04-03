import openai
import numpy as np
from typing import List, Union
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIInterface:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI interface with retry logic and timeout

        Args:
            api_key: OpenAI API key
            model: Model to use for embeddings
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        openai.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for one or more texts using OpenAI's API

        Args:
            texts: Single text or list of texts to embed

        Returns:
            List of embeddings as float arrays
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts,
                timeout=self.timeout
            )
            
            # Extract embeddings from response
            embeddings = [data['embedding'] for data in response['data']]
            return embeddings

        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Single embedding as float array
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        if self.model == "text-embedding-ada-002":
            return 1536
        raise ValueError(f"Unknown embedding dimension for model: {self.model}")