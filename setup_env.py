#!/usr/bin/env python3
"""
SignLink Environment Setup Script

This script ensures all required dependencies and data files are properly installed.
Run this before first launch or after environment changes.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def install_packages():
    """Install required packages with proper dependency resolution"""
    packages = [
        "numpy==1.23.5",  # Specific version to avoid compatibility issues
        "opencv-python",
        "tensorflow",
        "keras",
        "Pillow",
        "scikit-learn",
        "nltk",
        "spacy",
        "python-Levenshtein",
        "mediapipe"
    ]

    logger.info("Installing core dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    except subprocess.CalledProcessError as e:
        logger.error(f"Package installation failed: {e}")
        sys.exit(1)

def setup_nltk():
    """Download NLTK data"""
    try:
        import nltk
        nltk.download('words')
        nltk.download('brown')
        nltk.download('punkt')
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"NLTK setup failed: {e}")
        sys.exit(1)

def setup_spacy():
    """Install spaCy model"""
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            logger.info("spaCy model already installed")
        except OSError:
            logger.info("Downloading spaCy model...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ])
    except Exception as e:
        logger.error(f"spaCy setup failed: {e}")
        sys.exit(1)

def create_directories():
    """Create required directory structure"""
    dirs = [
        'data/raw',
        'data/processed/train',
        'data/processed/test',
        'assets/models',
        'assets/icons',
        'logs'
    ]

    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            sys.exit(1)

def main():
    logger.info("Starting SignLink environment setup...")

    # Run setup steps
    install_packages()
    setup_nltk()
    setup_spacy()
    create_directories()

    logger.info("\nSetup completed successfully!")
    logger.info("You can now run the application with:\n")
    logger.info("python main.py  # For GUI version")
    logger.info("python infer.py # For command-line version")

if __name__ == "__main__":
    main()
