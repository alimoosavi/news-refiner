from datetime import datetime
from typing import TypedDict


class News(TypedDict):
    source: str
    content: str
    published_date: datetime
