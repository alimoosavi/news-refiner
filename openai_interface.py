from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from typing import List, Union
import logging
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIInterface:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        timeout: int = 30,
        max_retries: int = 3,
        chunk_size: int = 1000,
        embedding_ctx_length: int = 8191
    ):
        """
        Initialize OpenAI interface using LangChain
        
        Args:
            api_key: OpenAI API key
            model: Model name for embeddings
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            chunk_size: Batch size for embedding requests
            embedding_ctx_length: Maximum context length
        """
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            chunk_size=chunk_size,
            max_retries=max_retries,
            timeout=timeout,
            embedding_ctx_length=embedding_ctx_length
        )
        self.logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for one or more texts using LangChain
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            List of embeddings as float arrays
        """
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Track token usage with LangChain's callback
            with get_openai_callback() as cb:
                embeddings = self.embeddings.embed_documents(texts)
                self.logger.info(f"Token usage: {cb.total_tokens} tokens")
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
        try:
            with get_openai_callback() as cb:
                embedding = self.embeddings.embed_query(text)
                self.logger.info(f"Token usage: {cb.total_tokens} tokens")
                return embedding
        except Exception as e:
            self.logger.error(f"Error getting single embedding: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        if self.embeddings.model == "text-embedding-ada-002":
            return 1536
        raise ValueError(f"Unknown embedding dimension for model: {self.embeddings.model}")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return self.embeddings.client.get_num_tokens(text)