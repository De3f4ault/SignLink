import os
import shutil
from sklearn.model_selection import train_test_split

class DatasetManager:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def create_dataset(self, test_size=0.2):
        """Organize raw data into train/test sets"""
        classes = os.listdir(self.raw_dir)
        os.makedirs(os.path.join(self.processed_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, "test"), exist_ok=True)

        for class_name in classes:
            class_dir = os.path.join(self.raw_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Get all samples for this class
            samples = [f for f in os.listdir(class_dir)
                      if f.endswith('.jpg') or f.endswith('.png')]

            # Split into train/test
            train, test = train_test_split(samples, test_size=test_size)

            # Copy files to processed directory
            self._copy_files(class_name, train, "train")
            self._copy_files(class_name, test, "test")

    def _copy_files(self, class_name, files, subset):
        """Helper to copy files to target directory"""
        src_dir = os.path.join(self.raw_dir, class_name)
        dst_dir = os.path.join(self.processed_dir, subset, class_name)

        os.makedirs(dst_dir, exist_ok=True)

        for file in files:
            src = os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, file)
            shutil.copy2(src, dst)
