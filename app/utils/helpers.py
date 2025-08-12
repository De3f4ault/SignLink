import logging
import os
from typing import Tuple

def setup_logging(log_file="signlink.log", level=logging.INFO):
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def get_resource_path(relative_path: str) -> str:
    """Get absolute path to resource"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)
