from datetime import datetime

from pydantic import BaseModel


class RawNews(BaseModel):
    id: int
    source: str
    content: str
    published_date: datetime

    def dict(self, **kwargs):
        news_dict = super().dict(**kwargs)
        news_dict['published_date'] = self.published_date.isoformat()
        return news_dict


class ShortNews(BaseModel):
    source: str
    timestamp: datetime
    body: str


class NewsLink(BaseModel):
    source: str
    timestamp: datetime
    body: str
