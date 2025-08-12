#!/usr/bin/env python3
"""Initialize project folder structure"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from app.utils.helpers import ensure_dir

def main():
    # Create required directories
    directories = [
        "data/raw",
        "data/processed/train",
        "data/processed/test",
        "data/augmented",
        "assets/models",
        "assets/icons",
        "assets/styles",
        "docs",
        "tests"
    ]

    for directory in directories:
        ensure_dir(directory)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    main()
