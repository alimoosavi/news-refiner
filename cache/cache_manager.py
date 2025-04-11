import json
from datetime import datetime


class CacheManager:
    """Handles loading and saving a cache of string keys and datetime values to/from a file."""

    def __init__(self, file_path):
        """Initialize with the path to the file where the cache will be stored."""
        self.file_path = file_path
        self.cache = self.load_cache()

    def load_cache(self):
        """Load the cache from the file, deserializing datetime values."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                # Deserialize datetime values from strings
                for key, value in data.items():
                    data[key] = datetime.fromisoformat(value)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            # Return an empty dictionary if the file doesn't exist or is corrupted
            return {}

    def save_cache(self):
        """Save the cache to the file, serializing datetime values to strings."""
        with open(self.file_path, 'w') as f:
            # Serialize datetime values to ISO format strings
            data_to_save = {key: value.isoformat() for key, value in self.cache.items()}
            json.dump(data_to_save, f, indent=4)

    def get(self, key):
        """Get a value from the cache."""
        return self.cache.get(key)

    def set(self, key, value):
        """Set a value in the cache, where value is a datetime object."""
        if not isinstance(value, datetime):
            raise ValueError("The value must be a datetime object.")
        self.cache[key] = value
        self.save_cache()

    def remove(self, key):
        """Remove a key from the cache."""
        if key in self.cache:
            del self.cache[key]
            self.save_cache()
