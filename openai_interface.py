import openai


class OpenAIInterface:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """
        Initializes the OpenAI interface with the provided API key and model.

        :param api_key: The OpenAI API key.
        :param model: The model to use for embeddings. Default is "text-embedding-ada-002".
        """
        openai.api_key = api_key
        self.model = model

    def get_embeddings(self, news_list: list) -> list:
        """
        Fetches the embedding vectors for a list of news articles in a single request.

        :param news_list: A list of news articles (titles + bodies) to get embeddings for.
        :return: A list of embedding vectors corresponding to the input news articles.
        """
        try:
            # Sending a batch request to OpenAI API
            response = openai.Embedding.create(
                input=news_list,
                model=self.model
            )
            # Extracting the embeddings from the response
            embeddings = [item['embedding'] for item in response['data']]
            return embeddings
        except Exception as e:
            print(f"Error fetching embeddings: {e}")
            return []