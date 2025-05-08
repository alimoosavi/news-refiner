import os
from celery import Celery
from datetime import timedelta
from config import config

# Set environment variable to use 'spawn' instead of 'fork'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Use environment variables for broker and backend URLs if available
broker_url = os.getenv('CELERY_BROKER_URL', config.celery.broker_url)
result_backend = os.getenv('CELERY_RESULT_BACKEND', config.celery.result_backend)

# Initialize Celery app
app = Celery('news_refiner', 
             broker=broker_url,
             backend=result_backend)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_expires=3600,
    worker_concurrency=config.celery.concurrency,
    task_time_limit=config.celery.task_time_limit,
    worker_pool='solo',  # Changed from 'prefork' to 'solo' for macOS compatibility
    beat_schedule={
        'process-news': {
            'task': 'news_embedding_pipeline_task.process_news_embeddings',
            'schedule': timedelta(minutes=1),
            'kwargs': {'limit': config.processing.batch_size}
        },
        'collect-news': {
            'task': 'news_collector_task.collect_news',
            'schedule': timedelta(minutes=1)
        },
    }
)

# Import tasks directly instead of using autodiscover
from news_collector_task import collect_news
from news_embedding_pipeline_task import process_news_embeddings

# Register tasks manually
app.tasks.register(collect_news)
app.tasks.register(process_news_embeddings)