#!/usr/bin/env python3
"""
Enhanced Dataset Ingestion Pipeline with:
- Comprehensive error handling
- Detailed progress reporting
- Memory optimizations
- Distributed processing support
"""

import os
import cv2
import json
import pandas as pd
import hashlib
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import logging
import multiprocessing
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

class EnhancedDatasetIngestor:
    def __init__(self, config: Dict):
        """
        Initialize with comprehensive configuration

        Args:
            config: {
                "source_dirs": List of directories to process,
                "target_root": Output directory,
                "image_size": (width, height),
                "supported_formats": [".jpg", ...],
                "class_mapping": {"del": "DELETE", ...},
                "validation_split": 0.2,
                "test_split": 0.1,
                "max_workers": Optional[int],
                "log_level": "INFO"
            }
        """
        self.config = config
        self.validate_config()

        # Initialize directories
        self.target_root = Path(config['target_root'])
        self.target_root.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Track processing state
        self.processed_hashes = set()
        self.load_hashes()

        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'total_discovered': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'classes': defaultdict(int),
            'formats': defaultdict(int),
            'json_files_processed': 0
        }

    def validate_config(self):
        """Ensure required config values are present"""
        required = ['source_dirs', 'target_root', 'image_size']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Set defaults
        self.config.setdefault('supported_formats', ['.jpg', '.jpeg', '.png'])
        self.config.setdefault('class_mapping', {})
        self.config.setdefault('validation_split', 0.2)
        self.config.setdefault('test_split', 0.1)
        self.config.setdefault('max_workers', multiprocessing.cpu_count() * 2)
        self.config.setdefault('log_level', 'INFO')
        self.config.setdefault('json_output_dir', 'json_metadata')

    def setup_logging(self):
        """Configure logging with file and console output"""
        logging.basicConfig(
            level=self.config['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.target_root / 'ingestion.log'),
                logging.StreamHandler()
            ]
        )

    def load_hashes(self):
        """Load previously processed file hashes"""
        hash_file = self.target_root / 'processed_hashes.txt'
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                self.processed_hashes = set(line.strip() for line in f)

    def save_hashes(self):
        """Save current hashes to disk"""
        with open(self.target_root / 'processed_hashes.txt', 'w') as f:
            f.write('\n'.join(self.processed_hashes))

    def run_pipeline(self):
        """Execute full processing pipeline"""
        self.logger.info("="*80)
        self.logger.info("Starting Enhanced Dataset Ingestion Pipeline")
        self.logger.info(f"Source directories: {self.config['source_dirs']}")
        self.logger.info(f"Target directory: {self.target_root}")
        self.logger.info("="*80)

        try:
            # Phase 1: File Discovery
            self.logger.info("\n[PHASE 1] Discovering files...")
            all_files = self.discover_files()
            self.logger.debug(f"Sample discovered files: {all_files[:5]}")

            # Phase 2: File Processing
            self.logger.info("\n[PHASE 2] Processing files...")
            self.process_files(all_files)

            # Phase 3: Data Organization
            self.logger.info("\n[PHASE 3] Organizing data...")
            self.organize_data()

            # Final Reporting
            self.generate_report()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
        finally:
            self.cleanup()

    def discover_files(self) -> List[Path]:
        """Find all supported files with detailed progress tracking"""
        discovered_files = []
        valid_sources = []

        # Validate source directories
        for src in self.config['source_dirs']:
            src_path = Path(src)
            if not src_path.exists():
                self.logger.warning(f"Source directory not found: {src_path}")
                continue
            if not src_path.is_dir():
                self.logger.warning(f"Source path is not a directory: {src_path}")
                continue
            valid_sources.append(src_path)

        if not valid_sources:
            raise ValueError("No valid source directories found")

        # Discover files with parallel processing
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []

            # Submit discovery tasks
            for src in valid_sources:
                self.logger.info(f"Discovering files in: {src}")
                futures.append(executor.submit(self._discover_in_directory, src))

            # Process results with progress bar
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Discovering files",
                unit="dir"
            ) as pbar:
                for future in pbar:
                    try:
                        files = future.result()
                        discovered_files.extend(files)
                        self.stats['total_discovered'] += len(files)
                        pbar.set_postfix({"found": len(discovered_files)})
                    except Exception as e:
                        self.logger.error(f"Discovery error: {str(e)}", exc_info=True)

        self.logger.info(f"Discovered {len(discovered_files)} total files")
        return discovered_files

    def _discover_in_directory(self, directory: Path) -> List[Path]:
        """Discover supported files in a single directory"""
        found_files = []

        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    ext = file_path.suffix.lower()

                    if ext in self.config['supported_formats']:
                        try:
                            file_hash = self._calculate_file_hash(file_path)

                            if file_hash in self.processed_hashes:
                                self.stats['skipped'] += 1
                                continue

                            found_files.append(file_path)
                            self.processed_hashes.add(file_hash)
                            self.stats['formats'][ext] += 1
                        except Exception as e:
                            self.stats['errors'] += 1
                            self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Error discovering files in {directory}: {str(e)}", exc_info=True)

        return found_files

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate file hash efficiently"""
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):  # 64KB chunks
                h.update(chunk)
        return h.hexdigest()

    def process_files(self, file_list: List[Path]):
        """Process files with detailed progress tracking"""
        if not file_list:
            self.logger.warning("No files to process")
            return

        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []

            # Submit all files for processing
            for file in file_list:
                futures.append(executor.submit(self._process_single_file, file))

            # Track progress with detailed updates
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing files",
                unit="file",
                mininterval=1.0  # Update at least once per second
            ) as pbar:
                for future in pbar:
                    try:
                        result = future.result()
                        if result:
                            self.stats['processed'] += 1
                            class_name = result[1] if isinstance(result, tuple) else "metadata"
                            self.stats['classes'][class_name] += 1
                            pbar.set_postfix({
                                "processed": self.stats['processed'],
                                "current": class_name[:10] + ("..." if len(class_name) > 10 else "")
                            })
                    except Exception as e:
                        self.stats['errors'] += 1
                        self.logger.error(f"Processing failed: {str(e)}", exc_info=True)

        self.logger.info(f"Successfully processed {self.stats['processed']} files")

    def _process_single_file(self, filepath: Path) -> Optional[Tuple[Path, str]]:
        """Process a single file with comprehensive error handling"""
        try:
            ext = filepath.suffix.lower()

            if ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
                return self._process_image(filepath)
            elif ext == '.csv':
                return self._process_csv(filepath)
            elif ext == '.json':
                return self._process_json(filepath)
            else:
                self.logger.warning(f"Unsupported file format: {filepath}")
                return None

        except Exception as e:
            raise Exception(f"Failed to process {filepath}: {str(e)}")

    def _process_image(self, img_path: Path) -> Tuple[Path, str]:
        """Process an image file with OpenCV"""
        try:
            # Read and validate image
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Invalid image file")

            # Standardize image format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.config['image_size'])

            # Determine class name
            class_name = self._extract_class_name(img_path)
            standardized_class = self.config['class_mapping'].get(class_name, class_name)

            # Save processed image
            output_dir = self.target_root / 'unified' / standardized_class
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{self._calculate_file_hash(img_path)}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            return output_path, standardized_class

        except Exception as e:
            raise Exception(f"Image processing failed: {str(e)}")

    def _process_json(self, json_path: Path) -> Optional[Tuple[Path, str]]:
        """Process a JSON file and save metadata"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Create output directory for JSON metadata
            output_dir = self.target_root / self.config['json_output_dir']
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save processed JSON with standardized name
            output_path = output_dir / f"{json_path.stem}_processed.json"

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.stats['json_files_processed'] += 1
            return output_path, "metadata"

        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON file: {str(e)}")
        except Exception as e:
            raise Exception(f"JSON processing failed: {str(e)}")

    def _process_csv(self, csv_path: Path) -> Optional[Tuple[Path, str]]:
        """Process a CSV file and save cleaned version"""
        try:
            df = pd.read_csv(csv_path)

            # Create output directory for CSV files
            output_dir = self.target_root / 'processed_csv'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save processed CSV
            output_path = output_dir / f"{csv_path.stem}_processed.csv"
            df.to_csv(output_path, index=False)

            return output_path, "tabular_data"

        except Exception as e:
            raise Exception(f"CSV processing failed: {str(e)}")

    def _extract_class_name(self, filepath: Path) -> str:
        """Extract class name from file path with multiple strategies"""
        # Strategy 1: Use parent directory name
        class_name = filepath.parent.name

        # Strategy 2: For nested datasets, look deeper
        parts = filepath.parts
        if 'asl_alphabet_train' in parts:
            # Handle ASL Alphabet dataset structure
            class_name = parts[parts.index('asl_alphabet_train') + 2]
        elif 'MS-ASL' in parts:
            # Handle MS-ASL dataset structure
            class_name = parts[parts.index('MS-ASL') + 2]

        # Apply any class name mappings
        return self.config['class_mapping'].get(class_name.lower(), class_name).upper()

    def organize_data(self):
        """Balance classes and create dataset splits"""
        self.logger.info("Balancing classes and creating splits...")

        try:
            unified_dir = self.target_root / 'unified'
            if not unified_dir.exists():
                raise ValueError("No processed files found in unified directory")

            # Prepare data for balancing
            class_counts = {}
            for class_dir in unified_dir.iterdir():
                if class_dir.is_dir():
                    samples = list(class_dir.glob('*.*'))
                    class_counts[class_dir.name] = len(samples)

            if not class_counts:
                raise ValueError("No classes found for balancing")

            # Balance classes
            ros = RandomOverSampler(random_state=42)
            X = np.array(list(class_counts.keys())).reshape(-1, 1)
            y = np.array(list(class_counts.keys()))

            # Create balanced distribution
            X_res, _ = ros.fit_resample(X, y)

            # Create dataset splits
            for class_name in np.unique(X_res):
                samples = list((unified_dir / class_name).glob('*.*'))

                # First split: train vs temp (val+test)
                train, temp = train_test_split(
                    samples,
                    test_size=(self.config['validation_split'] + self.config['test_split']),
                    random_state=42
                )

                # Second split: val vs test
                val, test = train_test_split(
                    temp,
                    test_size=self.config['test_split']/(self.config['validation_split'] + self.config['test_split']),
                    random_state=42
                )

                # Copy files to final directories
                for split_name, split_files in [('train', train), ('val', val), ('test', test)]:
                    dest_dir = self.target_root / split_name / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    for src_file in split_files:
                        dest_file = dest_dir / src_file.name
                        if not dest_file.exists():
                            shutil.copy2(src_file, dest_file)

            self.logger.info("Successfully created balanced dataset splits")

        except Exception as e:
            self.logger.error(f"Data organization failed: {str(e)}", exc_info=True)
            raise

    def generate_report(self):
        """Generate comprehensive processing report"""
        report = [
            "\n" + "="*80,
            "DATASET INGESTION REPORT",
            "="*80,
            f"\nProcessing started: {self.stats['start_time']}",
            f"Processing completed: {datetime.now()}",
            f"Total runtime: {datetime.now() - self.stats['start_time']}",

            "\nSUMMARY STATISTICS:",
            f"- Total files discovered: {self.stats['total_discovered']}",
            f"- Files successfully processed: {self.stats['processed']}",
            f"- Files skipped (already processed): {self.stats['skipped']}",
            f"- Errors encountered: {self.stats['errors']}",
            f"- JSON metadata files processed: {self.stats['json_files_processed']}",

            "\nFILE FORMATS PROCESSED:"
        ]

        for ext, count in sorted(self.stats['formats'].items()):
            report.append(f"- {ext}: {count} files")

        report.append("\nCLASS DISTRIBUTION:")
        for cls, count in sorted(self.stats['classes'].items()):
            report.append(f"- {cls}: {count} samples")

        # Save report
        report_path = self.target_root / 'ingestion_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        # Print to console
        self.logger.info('\n'.join(report))

    def cleanup(self):
        """Final cleanup tasks"""
        self.save_hashes()
        self.logger.info("Pipeline completed. Hashes saved for future runs.")

# Example Configuration
CONFIG = {
    "source_dirs": [
        "/home/de3f4ault/Desktop/Projects/Datasets/asl_alphabet_train/asl_alphabet_train",
        "/home/de3f4ault/Desktop/Projects/Datasets/asl_dataset",
        "/home/de3f4ault/Desktop/Projects/Datasets/MS-ASL"
    ],
    "target_root": "data/processed",
    "image_size": (128, 128),
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".csv", ".json"],
    "class_mapping": {
        "del": "DELETE",
        "space": "SPACE",
        "nothing": "BACKGROUND"
    },
    "validation_split": 0.15,
    "test_split": 0.15,
    "max_workers": multiprocessing.cpu_count() * 2,
    "log_level": "INFO",
    "json_output_dir": "json_metadata"
}

if __name__ == "__main__":
    try:
        ingestor = EnhancedDatasetIngestor(CONFIG)
        ingestor.run_pipeline()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)
