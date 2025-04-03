import asyncio

from telethon import TelegramClient

from collectors.schema import News


class ConcurrentTelegramChannelsCollector:
    """Fetch messages from multiple Telegram channels using a single Telethon session."""

    TELEGRAM_CHANNELS = {
        'IRNA': '@irna_1313',
        'ISNA': '@isna94',
        'FARS': '@farsna',
        'JAHAN_FOURI': '@Jahan_Fouri'
    }

    def __init__(self, api_id, api_hash, logger, session_name="collector_session"):
        """Initialize the Telegram client with the provided API credentials and logger."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.logger = logger

    async def start(self):
        """Start the Telethon client."""
        self.logger.info("Starting the Telegram client...")
        await self.client.start()

    async def stop(self):
        """Stop the Telethon client."""
        self.logger.info("Stopping the Telegram client...")
        await self.client.disconnect()

    async def fetch_historical_messages(self, source_name, channel, limit=10_000):
        """Fetch historical messages from a specific channel."""
        news = []
        try:
            self.logger.info(f"Fetching messages from {channel} (source: {source_name})...")
            async for message in self.client.iter_messages(channel, limit=limit):
                content = message.text if isinstance(message.text, str) else "No content available"
                if content:
                    news.append(News(source=source_name,
                                     content=content,
                                     published_date=message.date))
        except Exception as e:
            self.logger.error(f"Failed to fetch messages from {channel}: {e}")
            return []
        self.logger.info(f"Fetched {len(news)} messages from {channel}.")
        return news

    async def fetch_selected_channels(self, limit=100):
        """Fetch messages concurrently for selected channels using a single session."""
        self.logger.info("Fetching messages from selected Telegram channels...")

        # Create tasks for fetching messages from the selected channels
        tasks = {source_name: self.fetch_historical_messages(source_name, channel, limit)
                 for source_name, channel in self.TELEGRAM_CHANNELS.items()}

        # Execute all tasks concurrently and collect results
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Log any errors
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching messages from {key}: {result}")

        # Map results back to their respective channel keys
        fetched_data = {key: result if isinstance(result, list) else [] for key, result in zip(tasks.keys(), results)}
        self.logger.info("Completed fetching messages from all selected channels.")

        return fetched_data
