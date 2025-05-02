import os
from celery import Celery
from datetime import timedelta
from config import config

# Set environment variable to use 'spawn' instead of 'fork'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Initialize Celery app
app = Celery('news_refiner', broker=config.celery.broker_url)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_expires=3600,
    worker_concurrency=1,
    task_time_limit=config.celery.task_time_limit,
    worker_pool='solo',  # Use 'solo' pool to avoid forking issues on macOS
    beat_schedule={
        # 'collect-news-every-interval': {
        #     'task': 'news_collector_task.collect_news',
        #     'schedule': timedelta(minutes=1),
        # },
        'process-news': {
            'task': 'news_embedding_pipeline_task.process_news_embeddings',
            'schedule': timedelta(minutes=1),
            'kwargs': {'limit': 5}
        }
    }
)

# Import tasks modules
app.autodiscover_tasks([
    # 'news_collector_task',
    'news_embedding_pipeline_task'
])