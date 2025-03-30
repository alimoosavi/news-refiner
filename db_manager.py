from typing import List

import mysql.connector

from schema import RawNews, ShortNews


class DBManager:
    def __init__(self, db_name, user, password, host, port):
        self.conn = mysql.connector.connect(
            database=db_name, user=user, password=password, host=host, port=port
        )
        self.create_tables()  # Ensure tables exist on initialization

    def create_tables(self):
        """Create `short_news`, `long_news`, and `news_link` tables if they do not exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS short_news (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255),
                timestamp TIMESTAMP,
                title TEXT,
                body TEXT,
                has_processed BOOLEAN DEFAULT FALSE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS long_news (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255),
                timestamp TIMESTAMP,
                news_link TEXT UNIQUE,
                title TEXT,
                body TEXT,
                has_processed BOOLEAN DEFAULT FALSE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS news_link (
                id SERIAL PRIMARY KEY,
                source VARCHAR(255),
                timestamp TIMESTAMP,
                link VARCHAR(255) UNIQUE,
                has_processed BOOLEAN DEFAULT FALSE
            );
            """
        ]
        with self.conn.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
            self.conn.commit()

    def get_unprocessed_news(self):
        """Retrieve all unprocessed news articles."""
        with self.conn.cursor(dictionary=True) as cursor:
            cursor.execute("SELECT * FROM raw_news WHERE has_processed = FALSE;")
            data_list = cursor.fetchall()
            return [RawNews(**item) for item in data_list]

    def mark_news_as_processed(self, news_ids: List[int]):
        """Mark multiple news articles as processed."""
        if not news_ids:
            return  # No IDs provided, nothing to update

        with self.conn.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(news_ids))
            cursor.execute(f"""
                UPDATE raw_news SET has_processed = TRUE WHERE id IN ({format_strings});
            """, tuple(news_ids))
            self.conn.commit()

    def bulk_insert_short_news(self, news_list: List[ShortNews]):
        """Insert multiple ShortNews records in bulk into the database."""
        if not news_list:
            return  # No data to insert

        values = [(news.source, news.timestamp, news.title, news.body) for news in news_list]

        query = """
            INSERT INTO short_news (source, timestamp, title, body)
            VALUES (%s, %s, %s, %s);
        """

        with self.conn.cursor() as cursor:
            cursor.executemany(query, values)
            self.conn.commit()

    def process_and_insert_news(self, news_list: List[ShortNews], raw_news_ids: List[int]):
        """
        Atomically insert multiple ShortNews records and mark corresponding raw_news as processed.
        """
        if not news_list or not raw_news_ids:
            return  # Nothing to process

        insert_query = """
            INSERT INTO short_news (source, timestamp, title, body)
            VALUES (%s, %s, %s, %s);
        """

        update_query = f"""
            UPDATE raw_news SET has_processed = TRUE WHERE id IN ({','.join(['%s'] * len(raw_news_ids))});
        """

        values = [(news.source, news.timestamp, news.title, news.body) for news in news_list]

        try:
            with self.conn.cursor() as cursor:
                # Bulk insert short news
                cursor.executemany(insert_query, values)
                # Mark news as processed
                cursor.execute(update_query, tuple(raw_news_ids))
                # Commit transaction
                self.conn.commit()
        except mysql.connector.Error as e:
            self.conn.rollback()  # Rollback if any issue occurs
            raise e  # Re-raise the exception for proper error handling

    def get_raw_news_by_source_and_status(self, source: str, has_processed: bool) -> List[RawNews]:
        """Retrieve RawNews articles filtered by source and processed status."""
        with self.conn.cursor(dictionary=True) as cursor:
            cursor.execute("""
                SELECT * FROM raw_news
                WHERE source = %s AND has_processed = %s;
            """, (source, has_processed))
            data_list = cursor.fetchall()
            return [RawNews(**item) for item in data_list]

    def close(self):
        """Close the database connection."""
        self.conn.close()
