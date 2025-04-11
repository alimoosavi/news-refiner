from typing import List, Dict

import httpx


class NewsCrawlerClient:
    def __init__(self, base_url: str):
        """Initialize the client with the FastAPI server's base URL."""
        self._base_url = base_url

    async def fetch_news(self, source, links: List[str]) -> Dict[str, Dict[str, str]]:
        """Send a request to fetch news content asynchronously."""
        url = f"{self._base_url}/fetch_news/{source.lower()}/"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"links": links})

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Error: {response.status_code}, {response.text}")
