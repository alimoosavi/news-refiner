from datetime import datetime
import secrets
import string
from pydantic import BaseModel, Field


class RawNews(BaseModel):
    id: int
    source: str
    content: str
    published_date: datetime

    def dict(self, **kwargs):
        news_dict = super().dict(**kwargs)
        news_dict['published_date'] = self.published_date.isoformat()
        return news_dict


def generate_random_id(length: int = 20) -> str:
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


class ShortNews(BaseModel):
    id_: str = Field(default_factory=generate_random_id)
    source: str
    timestamp: datetime
    title: str
    body: str


class NewsLink(BaseModel):
    source: str
    timestamp: datetime
    body: str
