from config import config
from db.db_manager import DBManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('database_migration')

def main():
    # Create DB manager instance using configuration
    db_manager = DBManager(
        user=config.database.user,
        password=config.database.passkey,
        host=config.database.hostname,
        port=config.database.port,
        database=config.database.name,
        connector=config.database.connector,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        logger=logger
    )
    try:
        logger.info("Starting database initialization...")
        db_manager.initialize_database()
        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        db_manager.close_connections()

if __name__ == "__main__":
    main()