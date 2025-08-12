#!/usr/bin/env python3
"""
Enhanced Production-Ready Sign Language Training Pipeline
Following industry best practices with advanced monitoring and visualization
"""

import os
import gc
import sys
import logging
import argparse
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Tuple, Dict, List, Optional, Generator
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Conv2D, SeparableConv2D, Add, Multiply, GlobalMaxPooling2D,
    Concatenate, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
    TensorBoard, CSVLogger, LearningRateScheduler
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Suppress specific TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 1: all, 2: no info, 3: no warnings
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize colorama
colorama.init(autoreset=True)

# ======================== ASCII ART & STYLING ========================
class DisplayManager:
    """Enhanced display manager with ASCII art and progress tracking"""

    @staticmethod
    def print_banner():
        banner = f"""
{Fore.CYAN}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██╗ ██████╗ ███╗   ██╗    ██╗     ██╗███╗   ██╗██╗  ██╗           ║
║   ██╔════╝██║██╔════╝ ████╗  ██║    ██║     ██║████╗  ██║██║ ██╔╝           ║
║   ███████╗██║██║  ███╗██╔██╗ ██║    ██║     ██║██╔██╗ ██║█████╔╝            ║
║   ╚════██║██║██║   ██║██║╚██╗██║    ██║     ██║██║╚██╗██║██╔═██╗            ║
║   ███████║██║╚██████╔╝██║ ╚████║    ███████╗██║██║ ╚████║██║  ██╗           ║
║   ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚══════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝           ║
║                                                                              ║
║              Advanced Sign Language Recognition Training Pipeline            ║
║                           Powered by TensorFlow 2.x                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""
        print(banner)

    @staticmethod
    def print_section(title: str, icon: str = ">>"):
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}{icon} {title.upper()}")
        print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")

    @staticmethod
    def print_success(message: str):
        print(f"{Fore.GREEN}[SUCCESS] {message}{Style.RESET_ALL}")

    @staticmethod
    def print_info(message: str):
        print(f"{Fore.CYAN}[INFO] {message}{Style.RESET_ALL}")

    @staticmethod
    def print_warning(message: str):
        print(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}")

    @staticmethod
    def print_error(message: str):
        print(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}")

# ======================== CONFIGURATION ========================
class Config:
    """Enhanced configuration management with advanced settings"""

    # Environment
    TF_LOG_LEVEL = '2'  # Changed from '3' to allow info messages
    OMP_THREADS = '4'
    TF_ENABLE_GPU_MEMORY_GROWTH = 'true'

    # Model parameters
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32
    INITIAL_LR = 1e-3
    EPOCHS = 10
    NUM_CLASSES = 41

    # Advanced training parameters
    WARMUP_EPOCHS = 5
    COSINE_DECAY_STEPS = 1000
    LABEL_SMOOTHING = 0.1
    DROPOUT_RATE = 0.3
    L2_REGULARIZATION = 1e-4

    # Memory management
    MAX_MEMORY_USAGE = 0.75
    MIN_BATCH_SIZE = 8
    PREFETCH_BUFFER = tf.data.AUTOTUNE
    PARALLEL_CALLS = tf.data.AUTOTUNE

    # Paths
    DATA_DIR = Path('data/processed/train')
    MODEL_DIR = Path('assets/models')
    LOG_DIR = Path('logs')
    CHECKPOINT_DIR = Path('checkpoints')

    # Data augmentation (enhanced)
    AUGMENTATION_CONFIG = {
        'rotation_range': 15,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'brightness_range': [0.7, 1.3],
        'zoom_range': 0.15,
        'horizontal_flip': False,  # Sign language is directional
        'fill_mode': 'nearest',
        'shear_range': 0.1,
        'channel_shift_range': 20.0
    }

    # Model architecture options
    BACKBONE_OPTIONS = ['mobilenetv3', 'efficientnet']
    BACKBONE = 'mobilenetv3'
    USE_ATTENTION = True
    USE_MIXED_PRECISION = True

# ======================== ADVANCED CALLBACKS ========================
class CustomProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback with enhanced progress tracking"""

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_progress = None

    def on_train_begin(self, logs=None):
        DisplayManager.print_info("Starting training process...")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n{Fore.MAGENTA}[EPOCH {epoch + 1}/{self.total_epochs}]{Style.RESET_ALL}")
        self.epoch_progress = tqdm(
            total=self.params['steps'],
            desc=f"Training",
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='green'
        )

    def on_batch_end(self, batch, logs=None):
        if self.epoch_progress:
            self.epoch_progress.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_progress:
            self.epoch_progress.close()

        # Display metrics
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        lr = logs.get('lr', 0)

        print(f"{Fore.GREEN}Training   -> Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"{Fore.BLUE}Validation -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        print(f"{Fore.YELLOW}Learning Rate: {lr:.2e}{Style.RESET_ALL}")

# ======================== SYSTEM MONITORING ========================
class AdvancedSystemMonitor:
    """Enhanced system resource monitoring with real-time tracking"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and properly configured"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        return len(gpus) > 0

    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        return self.process.memory_info().rss / (1024 ** 2)

    def get_system_memory(self) -> Tuple[float, float, float]:
        """Get system memory stats: (available_gb, used_gb, percent_used)"""
        mem = psutil.virtual_memory()
        return (
            mem.available / (1024 ** 3),
            mem.used / (1024 ** 3),
            mem.percent
        )

    def get_gpu_info(self) -> Dict:
        """Get comprehensive GPU information"""
        try:
            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            if not gpu_devices:
                return {'available': False, 'count': 0}

            gpu_info = {
                'available': True,
                'count': len(gpu_devices),
                'devices': []
            }

            for i, device in enumerate(gpu_devices):
                device_info = {
                    'id': i,
                    'name': device.name,
                    'device_type': device.device_type
                }
                gpu_info['devices'].append(device_info)

            return gpu_info
        except Exception as e:
            return {'available': False, 'count': 0, 'error': str(e)}

    def print_system_info(self):
        """Print comprehensive system information"""
        DisplayManager.print_section("SYSTEM INFORMATION", "🖥️")

        # Python and TensorFlow versions
        DisplayManager.print_info(f"Python Version: {sys.version.split()[0]}")
        DisplayManager.print_info(f"TensorFlow Version: {tf.__version__} (includes Keras)")

        # Memory information
        mem_usage = self.get_memory_usage()
        available_mem, used_mem, mem_percent = self.get_system_memory()

        DisplayManager.print_info(f"Process Memory: {mem_usage:.1f} MB")
        DisplayManager.print_info(f"System Memory: {used_mem:.1f}/{available_mem + used_mem:.1f} GB ({mem_percent:.1f}%)")

        # GPU information
        gpu_info = self.get_gpu_info()
        if gpu_info['available']:
            DisplayManager.print_success(f"GPU Available: {gpu_info['count']} device(s)")
            for device in gpu_info['devices']:
                DisplayManager.print_info(f"  - GPU {device['id']}: {device['name']}")
        else:
            DisplayManager.print_warning("No GPU detected - using CPU")

    @contextmanager
    def memory_tracking(self, operation_name: str):
        """Enhanced memory tracking with progress indication"""
        start_memory = self.get_memory_usage()
        start_time = time.time()

        print(f"{Fore.CYAN}[MONITOR] Starting {operation_name}...{Style.RESET_ALL}")

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            duration = time.time() - start_time
            memory_delta = end_memory - start_memory

            status_color = Fore.GREEN if memory_delta < 100 else Fore.YELLOW if memory_delta < 500 else Fore.RED

            print(f"{status_color}[MONITOR] {operation_name} completed:")
            print(f"  Memory: {start_memory:.1f}MB -> {end_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
            print(f"  Duration: {duration:.2f}s{Style.RESET_ALL}")

# ======================== ENHANCED DATA PIPELINE ========================
class AdvancedDataPipeline:
    """High-performance data pipeline with advanced augmentation"""

    def __init__(self, monitor: AdvancedSystemMonitor):
        self.monitor = monitor
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.class_distribution = {}
        self.data_stats = {}

    def discover_dataset(self, data_dir: Path) -> Tuple[List[Path], List[str]]:
        """Enhanced dataset discovery with detailed analysis"""
        DisplayManager.print_section("DATASET DISCOVERY", "📁")

        if not data_dir.exists():
            raise ValueError(f"Dataset directory not found: {data_dir}")

        DisplayManager.print_info(f"Scanning directory: {data_dir}")

        # Find class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])

        DisplayManager.print_success(f"Found {len(class_names)} classes")

        # Initialize progress bar for dataset scanning
        progress = tqdm(class_dirs, desc="Scanning classes", colour="blue")

        image_paths = []
        class_counts = {}
        corrupted_counts = defaultdict(int)
        total_size = 0

        for class_dir in progress:
            class_name = class_dir.name
            progress.set_description(f"Scanning {class_name}")

            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

            class_images = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in valid_extensions
            ]

            # Validate images
            valid_images = []
            for img_path in class_images:
                try:
                    # Quick validity check
                    img = tf.io.read_file(str(img_path))
                    tf.image.decode_image(img)
                    valid_images.append(img_path)
                    total_size += img_path.stat().st_size
                except Exception:
                    corrupted_counts[class_name] += 1

            class_counts[class_name] = len(valid_images)
            image_paths.extend(valid_images)

        progress.close()

        # Display statistics
        DisplayManager.print_success(f"Dataset scan completed!")
        DisplayManager.print_info(f"Total valid images: {len(image_paths):,}")
        DisplayManager.print_info(f"Total dataset size: {total_size / (1024**3):.2f} GB")
        DisplayManager.print_info(f"Average images per class: {len(image_paths) / len(class_names):.1f}")

        if sum(corrupted_counts.values()) > 0:
            DisplayManager.print_warning(f"Corrupted images found: {sum(corrupted_counts.values())}")

        self.class_distribution = class_counts
        self.class_names = class_names
        self.data_stats = {
            'total_images': len(image_paths),
            'total_size_gb': total_size / (1024**3),
            'corrupted_count': sum(corrupted_counts.values())
        }

        return image_paths, class_names

    def analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        DisplayManager.print_section("CLASS DISTRIBUTION ANALYSIS", "📊")

        if not self.class_distribution:
            DisplayManager.print_warning("No class distribution data available")
            return

        # Calculate statistics
        counts = list(self.class_distribution.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = min(counts)
        max_count = max(counts)

        DisplayManager.print_info(f"Class count statistics:")
        DisplayManager.print_info(f"  Mean: {mean_count:.1f}")
        DisplayManager.print_info(f"  Std:  {std_count:.1f}")
        DisplayManager.print_info(f"  Min:  {min_count} ({list(self.class_distribution.keys())[counts.index(min_count)]})")
        DisplayManager.print_info(f"  Max:  {max_count} ({list(self.class_distribution.keys())[counts.index(max_count)]})")

        # Identify imbalanced classes
        threshold = mean_count * 0.5
        imbalanced_classes = [k for k, v in self.class_distribution.items() if v < threshold]

        if imbalanced_classes:
            DisplayManager.print_warning(f"Potentially imbalanced classes ({len(imbalanced_classes)}): {imbalanced_classes[:5]}...")

    def create_advanced_augmentation(self) -> tf.keras.Sequential:
        """Create advanced data augmentation pipeline that works on CPU"""
        # Determine if we're using mixed precision
        use_mixed_precision = Config.USE_MIXED_PRECISION and tf.config.list_physical_devices('GPU')

        # Create augmentation layers with appropriate dtype handling
        layers = [
            tf.keras.layers.RandomRotation(
                factor=Config.AUGMENTATION_CONFIG['rotation_range'] / 360.0,
                fill_mode=Config.AUGMENTATION_CONFIG['fill_mode']
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=Config.AUGMENTATION_CONFIG['height_shift_range'],
                width_factor=Config.AUGMENTATION_CONFIG['width_shift_range'],
                fill_mode=Config.AUGMENTATION_CONFIG['fill_mode']
            ),
            tf.keras.layers.RandomZoom(
                height_factor=Config.AUGMENTATION_CONFIG['zoom_range'],
                width_factor=Config.AUGMENTATION_CONFIG['zoom_range'],
                fill_mode=Config.AUGMENTATION_CONFIG['fill_mode']
            ),
        ]

        # Only add brightness/contrast if using GPU or not mixed precision
        if not use_mixed_precision:
            layers.extend([
                tf.keras.layers.RandomBrightness(
                    factor=0.2,
                    value_range=(0, 1)
                ),
                tf.keras.layers.RandomContrast(
                    factor=0.2
                )
            ])
        else:
            # Alternative augmentation for GPU with mixed precision
            layers.append(
                tf.keras.layers.Lambda(
                    lambda x: tf.image.adjust_brightness(x, delta=0.1),
                    name='brightness_adjust'
                )
            )

        return tf.keras.Sequential(layers)

    def preprocess_image(self, image_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Enhanced image preprocessing with normalization"""
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)

        # Convert to float32 first (required for CPU operations)
        image = tf.cast(image, tf.float32)

        # Resize with anti-aliasing
        image = tf.image.resize(image, Config.IMAGE_SIZE, method='lanczos3', antialias=True)

        # Normalize to [0, 1] range
        image = image / 255.0

        # Convert to mixed precision if enabled and GPU available
        if Config.USE_MIXED_PRECISION and tf.config.list_physical_devices('GPU'):
            image = tf.cast(image, tf.float16)

        # One-hot encode label
        label = tf.one_hot(label, depth=Config.NUM_CLASSES)

        return image, label

    def create_tf_dataset(self, image_paths: List[Path], labels: List[str], is_training: bool = True) -> tf.data.Dataset:
        """Create optimized tf.data pipeline with advanced features"""

        # Convert paths and encode labels
        path_strings = [str(p) for p in image_paths]
        encoded_labels = self.label_encoder.fit_transform(labels) if is_training else self.label_encoder.transform(labels)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((path_strings, encoded_labels))

        # Shuffle for training
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=min(10000, len(path_strings)),
                seed=42,
                reshuffle_each_iteration=True
            )

        # Preprocessing
        dataset = dataset.map(
            self.preprocess_image,
            num_parallel_calls=Config.PARALLEL_CALLS,
            deterministic=False
        )

        # Data augmentation for training
        if is_training:
            augment_layers = self.create_advanced_augmentation()
            dataset = dataset.map(
                lambda x, y: (augment_layers(x, training=True), y),
                num_parallel_calls=Config.PARALLEL_CALLS
            )

        # Batching
        dataset = dataset.batch(
            Config.BATCH_SIZE,
            drop_remainder=is_training,
            num_parallel_calls=Config.PARALLEL_CALLS
        )

        # Prefetching
        dataset = dataset.prefetch(Config.PREFETCH_BUFFER)

        return dataset

# ======================== ADVANCED MODEL ARCHITECTURE ========================
def create_attention_block(x, filters: int):
    """Create an attention mechanism block"""
    # Channel attention
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)

    # Shared MLP
    shared_mlp = tf.keras.Sequential([
        Dense(filters // 8, activation='relu'),
        Dense(filters, activation='sigmoid')
    ])

    channel_att = Add()([shared_mlp(gap), shared_mlp(gmp)])
    channel_att = tf.keras.layers.Reshape((1, 1, filters))(channel_att)
    x_channel = Multiply()([x, channel_att])

    return x_channel

def build_advanced_model(num_classes: int = Config.NUM_CLASSES) -> Model:
    """Build advanced model with attention mechanisms"""
    DisplayManager.print_section("MODEL ARCHITECTURE", "🏗️")

    # Base model selection
    if Config.BACKBONE == 'efficientnet':
        base_model = EfficientNetB0(
            input_shape=(*Config.IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        DisplayManager.print_info("Using EfficientNetB0 backbone")
    else:
        base_model = MobileNetV3Small(
            input_shape=(*Config.IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        DisplayManager.print_info("Using MobileNetV3Small backbone")

    # Custom head with attention
    x = base_model.output

    if Config.USE_ATTENTION:
        x = create_attention_block(x, x.shape[-1])
        DisplayManager.print_info("Attention mechanism enabled")

    # Global pooling
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    x = Concatenate()([gap, gmp])

    # Dense layers with advanced regularization
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(Config.DROPOUT_RATE)(x)

    x = Dense(512, activation='relu', kernel_regularizer=l2(Config.L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = Dropout(Config.DROPOUT_RATE * 0.7)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(Config.L2_REGULARIZATION))(x)
    x = Dropout(Config.DROPOUT_RATE * 0.5)(x)

    # Output layer with label smoothing support
    predictions = Dense(
        num_classes,
        activation='softmax',
        name='predictions',
        kernel_regularizer=l2(Config.L2_REGULARIZATION)
    )(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model initially for transfer learning
    base_model.trainable = False

    # Advanced optimizer with learning rate scheduling
    optimizer = AdamW(
        learning_rate=Config.INITIAL_LR,
        weight_decay=Config.L2_REGULARIZATION,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    # Compile with advanced metrics
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )

    # Model statistics
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    DisplayManager.print_success("Model architecture created successfully!")
    DisplayManager.print_info(f"Total parameters: {total_params:,}")
    DisplayManager.print_info(f"Trainable parameters: {trainable_params:,}")
    DisplayManager.print_info(f"Non-trainable parameters: {non_trainable_params:,}")
    DisplayManager.print_info(f"Model size (approx): {total_params * 4 / (1024**2):.1f} MB")

    return model

# ======================== ADVANCED TRAINING SETUP ========================
def create_advanced_callbacks(model_dir: Path, log_dir: Path, total_epochs: int) -> List:
    """Create comprehensive callback suite"""
    DisplayManager.print_section("TRAINING CALLBACKS", "🎯")

    callbacks = [
        # Custom progress callback
        CustomProgressCallback(total_epochs),

        # Enhanced model checkpointing
        ModelCheckpoint(
            filepath=str(model_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=0,
            save_freq='epoch'
        ),

        # Backup checkpoint
        ModelCheckpoint(
            filepath=str(model_dir / 'checkpoint_epoch_{epoch:03d}.keras'),
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,
            mode='min',
            verbose=0,
            save_freq=10  # Save every 10 epochs
        ),

        # Advanced learning rate scheduling
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-8,
            verbose=0,
            cooldown=5,
            min_delta=1e-4
        ),

        # Early stopping with restoration
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0,
            min_delta=1e-4,
            mode='min'
        ),

        # TensorBoard with comprehensive logging
        TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch='10,20',
            embeddings_freq=1
        ),

        # CSV logging
        CSVLogger(
            filename=str(log_dir / 'training_metrics.csv'),
            append=False,
            separator=','
        )
    ]

    DisplayManager.print_success(f"Created {len(callbacks)} advanced callbacks")
    return callbacks

def cosine_decay_with_warmup(epoch, lr):
    """Custom learning rate schedule with warmup"""
    if epoch < Config.WARMUP_EPOCHS:
        return Config.INITIAL_LR * (epoch + 1) / Config.WARMUP_EPOCHS
    else:
        progress = (epoch - Config.WARMUP_EPOCHS) / (Config.EPOCHS - Config.WARMUP_EPOCHS)
        return Config.INITIAL_LR * 0.5 * (1 + np.cos(np.pi * progress))

# ======================== ENVIRONMENT SETUP ========================
def setup_advanced_environment():
    """Setup optimized training environment"""
    DisplayManager.print_section("ENVIRONMENT SETUP", "⚙️")

    # Environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = Config.TF_LOG_LEVEL
    os.environ['OMP_NUM_THREADS'] = Config.OMP_THREADS
    os.environ['TF_ENABLE_GPU_MEMORY_GROWTH'] = Config.TF_ENABLE_GPU_MEMORY_GROWTH

    # Memory optimization
    tf.config.optimizer.set_jit(True)  # Enable XLA

    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            DisplayManager.print_success(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            DisplayManager.print_warning(f"GPU memory growth setup failed: {e}")

    # Mixed precision - only enable if GPU is available
    if Config.USE_MIXED_PRECISION:
        if gpus:
            try:
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                DisplayManager.print_success("Mixed precision (float16) enabled")
                return True
            except Exception as e:
                DisplayManager.print_warning(f"Mixed precision failed, using float32: {e}")
                return False
        else:
            DisplayManager.print_warning("Mixed precision disabled - no GPU detected")
            return False
    return False

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging system"""
    Config.LOG_DIR.mkdir(exist_ok=True, parents=True)

    # Create logger
    logger = logging.getLogger('SignLanguageTraining')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(
        Config.LOG_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

# ======================== MAIN TRAINING PIPELINE ========================
def train_advanced_model():
    """Advanced training pipeline with comprehensive monitoring"""
    start_time = time.time()

    # Display banner
    DisplayManager.print_banner()

    # Setup logging
    logger = setup_logging()

    try:
        # 1. Environment Setup
        mixed_precision_enabled = setup_advanced_environment()

        # 2. System Monitoring
        monitor = AdvancedSystemMonitor()
        monitor.print_system_info()

        # 3. Create directories
        for directory in [Config.MODEL_DIR, Config.LOG_DIR, Config.CHECKPOINT_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        # 4. Dataset Discovery and Analysis
        with monitor.memory_tracking("Dataset Discovery"):
            pipeline = AdvancedDataPipeline(monitor)
            image_paths, class_names = pipeline.discover_dataset(Config.DATA_DIR)
            pipeline.analyze_class_distribution()

        # 5. Train/Validation Split
        DisplayManager.print_section("DATA SPLITTING", "🔄")

        # Create labels from paths
        labels = [path.parent.name for path in image_paths]

        with monitor.memory_tracking("Data Split"):
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels,
                test_size=0.2,
                stratify=labels,
                random_state=42
            )

        DisplayManager.print_success(f"Training samples: {len(train_paths):,}")
        DisplayManager.print_success(f"Validation samples: {len(val_paths):,}")
        DisplayManager.print_info(f"Split ratio: {len(train_paths)/len(image_paths):.1%} train / {len(val_paths)/len(image_paths):.1%} val")

        # 6. Data Pipeline Creation
        DisplayManager.print_section("DATA PIPELINE CREATION", "🔄")

        with monitor.memory_tracking("Data Pipeline Creation"):
            # Create datasets
            DisplayManager.print_info("Creating training dataset...")
            train_dataset = pipeline.create_tf_dataset(train_paths, train_labels, is_training=True)

            DisplayManager.print_info("Creating validation dataset...")
            val_dataset = pipeline.create_tf_dataset(val_paths, val_labels, is_training=False)

        DisplayManager.print_success("Data pipelines created successfully!")
        DisplayManager.print_info(f"Batch size: {Config.BATCH_SIZE}")
        DisplayManager.print_info("Features: Advanced augmentation, prefetching, parallel processing")

        # 7. Class Weight Computation
        DisplayManager.print_section("CLASS BALANCING", "⚖️")

        with monitor.memory_tracking("Class Weight Computation"):
            # Compute class weights for imbalanced dataset
            unique_labels = np.unique(labels)
            class_weights = compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=labels
            )
            class_weight_dict = dict(zip(range(len(unique_labels)), class_weights))

        DisplayManager.print_success("Class weights computed for balanced training")
        weight_stats = f"Min: {min(class_weights):.3f}, Max: {max(class_weights):.3f}, Mean: {np.mean(class_weights):.3f}"
        DisplayManager.print_info(f"Weight statistics: {weight_stats}")

        # 8. Model Architecture
        with monitor.memory_tracking("Model Creation"):
            model = build_advanced_model(len(class_names))

        # 9. Training Configuration
        DisplayManager.print_section("TRAINING CONFIGURATION", "🎯")

        steps_per_epoch = len(train_paths) // Config.BATCH_SIZE
        validation_steps = len(val_paths) // Config.BATCH_SIZE

        DisplayManager.print_info(f"Steps per epoch: {steps_per_epoch:,}")
        DisplayManager.print_info(f"Validation steps: {validation_steps:,}")
        DisplayManager.print_info(f"Total epochs: {Config.EPOCHS}")
        DisplayManager.print_info(f"Estimated training time: {(steps_per_epoch * Config.EPOCHS * Config.BATCH_SIZE) / 3600:.1f} hours")

        # 10. Callback Setup
        callbacks = create_advanced_callbacks(Config.MODEL_DIR, Config.LOG_DIR, Config.EPOCHS)

        # Add learning rate scheduler
        lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup, verbose=0)
        callbacks.append(lr_scheduler)

        # 11. Training Execution
        DisplayManager.print_section("TRAINING EXECUTION", "🚀")

        print(f"{Fore.CYAN}Starting training process...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Monitor progress in TensorBoard: tensorboard --logdir {Config.LOG_DIR}{Style.RESET_ALL}")

        with monitor.memory_tracking("Model Training"):
            history = model.fit(
                train_dataset,
                epochs=Config.EPOCHS,
                validation_data=val_dataset,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=0,  # Suppress default verbose output (we have custom progress)
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )

        # 12. Post-Training Analysis
        DisplayManager.print_section("POST-TRAINING ANALYSIS", "📊")

        # Save final model
        final_model_path = Config.MODEL_DIR / 'final_model.keras'
        model.save(final_model_path)

        # Save additional artifacts
        np.save(Config.MODEL_DIR / 'class_names.npy', class_names)
        np.save(Config.MODEL_DIR / 'training_history.npy', history.history)

        # Training statistics
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        final_lr = history.history['lr'][-1]

        # Performance metrics
        training_time = time.time() - start_time
        final_memory = monitor.get_memory_usage()
        memory_delta = final_memory - monitor.initial_memory

        # Display final results
        DisplayManager.print_section("TRAINING COMPLETED", "🎉")

        results_table = f"""
{Fore.GREEN}╔══════════════════════════════════════════════════════════════╗
║                        TRAINING RESULTS                      ║
╠══════════════════════════════════════════════════════════════╣
║ Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch:3d})          ║
║ Best Validation Loss:     {best_val_loss:.4f}                       ║
║ Final Learning Rate:      {final_lr:.2e}                     ║
║ Total Training Time:      {training_time/3600:.2f} hours              ║
║ Memory Usage:             {monitor.initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)     ║
║ Model Size:               {final_model_path.stat().st_size/(1024**2):.1f} MB                    ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}"""

        print(results_table)

        # Save training summary
        summary = {
            'best_val_accuracy': float(best_val_acc),
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(best_epoch),
            'total_training_time': training_time,
            'final_memory_usage': final_memory,
            'memory_delta': memory_delta,
            'total_parameters': model.count_params(),
            'dataset_size': len(image_paths),
            'num_classes': len(class_names)
        }

        import json
        with open(Config.MODEL_DIR / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        DisplayManager.print_success("Training completed successfully!")
        DisplayManager.print_info(f"Model saved to: {final_model_path}")
        DisplayManager.print_info(f"Logs saved to: {Config.LOG_DIR}")
        DisplayManager.print_info(f"Summary saved to: {Config.MODEL_DIR / 'training_summary.json'}")

        return model, history

    except KeyboardInterrupt:
        DisplayManager.print_warning("Training interrupted by user")
        return None, None

    except Exception as e:
        DisplayManager.print_error(f"Training failed: {str(e)}")
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise

    finally:
        # Cleanup
        DisplayManager.print_section("CLEANUP", "🧹")
        DisplayManager.print_info("Performing cleanup operations...")

        gc.collect()
        if 'model' in locals():
            del model
        tf.keras.backend.clear_session()

        DisplayManager.print_success("Cleanup completed")

# ======================== FINE-TUNING FUNCTIONALITY ========================
def fine_tune_model(model: Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset,
                    steps_per_epoch: int, validation_steps: int) -> Model:
    """Fine-tune the model by unfreezing base layers"""
    DisplayManager.print_section("FINE-TUNING PHASE", "🔧")

    # Unfreeze the base model
    base_model = model.layers[0] if hasattr(model.layers[0], 'trainable') else None
    if base_model:
        base_model.trainable = True
        DisplayManager.print_info("Base model layers unfrozen for fine-tuning")

        # Use a lower learning rate for fine-tuning
        fine_tune_lr = Config.INITIAL_LR / 10
        model.compile(
            optimizer=AdamW(learning_rate=fine_tune_lr, weight_decay=Config.L2_REGULARIZATION),
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )

        DisplayManager.print_info(f"Fine-tuning learning rate: {fine_tune_lr:.2e}")

        # Fine-tuning epochs (fewer epochs with lower learning rate)
        fine_tune_epochs = min(20, Config.EPOCHS // 5)

        # Create callbacks for fine-tuning
        fine_tune_callbacks = [
            ModelCheckpoint(
                filepath=str(Config.MODEL_DIR / 'fine_tuned_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]

        DisplayManager.print_info(f"Starting fine-tuning for {fine_tune_epochs} epochs...")

        # Fine-tuning training
        fine_tune_history = model.fit(
            train_dataset,
            epochs=fine_tune_epochs,
            validation_data=val_dataset,
            callbacks=fine_tune_callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )

        DisplayManager.print_success("Fine-tuning completed!")
        return model
    else:
        DisplayManager.print_warning("No base model found for fine-tuning")
        return model

# ======================== EVALUATION UTILITIES ========================
def evaluate_model(model_path: Path, test_data_dir: Path):
    """Comprehensive model evaluation with detailed metrics"""
    DisplayManager.print_section("MODEL EVALUATION", "🔍")

    try:
        # Load model and metadata
        model = tf.keras.models.load_model(str(model_path))
        class_names = np.load(Config.MODEL_DIR / 'class_names.npy')

        DisplayManager.print_success(f"Model loaded from: {model_path}")
        DisplayManager.print_info(f"Classes: {len(class_names)}")

        # Load test data
        monitor = AdvancedSystemMonitor()
        pipeline = AdvancedDataPipeline(monitor)

        test_paths, _ = pipeline.discover_dataset(test_data_dir)
        test_labels = [path.parent.name for path in test_paths]

        # Create test dataset (no augmentation)
        test_dataset = pipeline.create_tf_dataset(test_paths, test_labels, is_training=False)

        # Evaluate with progress bar
        DisplayManager.print_info("Running comprehensive evaluation...")

        total_batches = len(test_paths) // Config.BATCH_SIZE

        with tqdm(total=total_batches, desc="Evaluating", colour="green") as pbar:
            # Get predictions
            predictions = []
            true_labels = []

            for batch_data, batch_labels in test_dataset:
                batch_preds = model.predict(batch_data, verbose=0)
                predictions.extend(np.argmax(batch_preds, axis=1))
                true_labels.extend(np.argmax(batch_labels, axis=1))
                pbar.update(1)

        # Calculate metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))

        # Display results
        DisplayManager.print_success(f"Test Accuracy: {accuracy:.4f}")

        # Generate detailed classification report
        report = classification_report(
            true_labels, predictions,
            target_names=class_names,
            output_dict=True
        )

        DisplayManager.print_info("Per-class performance:")
        for class_name in class_names[:10]:  # Show first 10 classes
            if class_name in report:
                metrics = report[class_name]
                DisplayManager.print_info(
                    f"  {class_name}: Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}"
                )

        # Save evaluation results
        eval_results = {
            'test_accuracy': float(accuracy),
            'classification_report': report,
            'total_test_samples': len(test_paths)
        }

        with open(Config.MODEL_DIR / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)

        DisplayManager.print_success("Evaluation completed and saved!")

    except Exception as e:
        DisplayManager.print_error(f"Evaluation failed: {str(e)}")
        raise

# ======================== PREDICTION UTILITIES ========================
def predict_single_image(model_path: Path, image_path: Path, class_names: List[str]):
    """Make prediction on a single image"""
    try:
        model = tf.keras.models.load_model(str(model_path))

        # Preprocess image
        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, Config.IMAGE_SIZE)
        image = image / 255.0
        image = tf.expand_dims(image, 0)

        # Make prediction
        prediction = model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]

        # Get top 3 predictions
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [(class_names[idx], prediction[0][idx]) for idx in top_3_idx]

        return {
            'predicted_class': class_names[predicted_class_idx],
            'confidence': float(confidence),
            'top_3': top_3_predictions
        }

    except Exception as e:
        DisplayManager.print_error(f"Prediction failed: {str(e)}")
        return None

# ======================== MAIN EXECUTION ========================
def main():
    """Enhanced main entry point with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description='🚀 Advanced Sign Language Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Fore.CYAN}📋 USAGE EXAMPLES:{Style.RESET_ALL}
  {Fore.GREEN}Basic Training:{Style.RESET_ALL}
    python train_model.py

  {Fore.GREEN}Custom Configuration:{Style.RESET_ALL}
    python train_model.py --epochs 50 --batch-size 64 --backbone efficientnet

  {Fore.GREEN}Model Evaluation:{Style.RESET_ALL}
    python train_model.py --evaluate assets/models/best_model.keras --test-data data/test

  {Fore.GREEN}Single Image Prediction:{Style.RESET_ALL}
    python train_model.py --predict image.jpg --model assets/models/best_model.keras

{Fore.YELLOW}📁 DIRECTORY STRUCTURE:{Style.RESET_ALL}
  data/processed/train/     # Training data (organized by class folders)
  assets/models/           # Saved models and artifacts
  logs/                   # Training logs and TensorBoard data
  checkpoints/            # Model checkpoints during training
        """
    )

    # Training arguments
    training_group = parser.add_argument_group('🏋️ Training Configuration')
    training_group.add_argument('--epochs', type=int, default=100,
                               help='Number of training epochs (default: 100)')
    training_group.add_argument('--batch-size', type=int, default=32,
                               help='Training batch size (default: 32)')
    training_group.add_argument('--learning-rate', type=float, default=1e-3,
                               help='Initial learning rate (default: 1e-3)')
    training_group.add_argument('--fine-tune', action='store_true',
                               help='Enable fine-tuning after initial training')

    # Model arguments
    model_group = parser.add_argument_group('🏗️ Model Architecture')
    model_group.add_argument('--backbone', choices=['mobilenetv3', 'efficientnet'],
                            default='mobilenetv3',
                            help='Model backbone architecture (default: mobilenetv3)')
    model_group.add_argument('--no-attention', action='store_true',
                            help='Disable attention mechanism')
    model_group.add_argument('--no-mixed-precision', action='store_true',
                            help='Disable mixed precision training')
    model_group.add_argument('--image-size', type=int, nargs=2, default=[128, 128],
                            help='Input image size [height width] (default: 128 128)')

    # Data arguments
    data_group = parser.add_argument_group('📊 Data Configuration')
    data_group.add_argument('--data-dir', type=Path, default=Path('data/processed/train'),
                           help='Training data directory')
    data_group.add_argument('--output-dir', type=Path, default=Path('assets/models'),
                           help='Model output directory')
    data_group.add_argument('--augment-strength', type=float, default=1.0,
                           help='Data augmentation strength multiplier (default: 1.0)')

    # Evaluation arguments
    eval_group = parser.add_argument_group('🔍 Evaluation & Prediction')
    eval_group.add_argument('--evaluate', type=Path,
                           help='Evaluate existing model (provide model path)')
    eval_group.add_argument('--test-data', type=Path,
                           help='Test data directory for evaluation')
    eval_group.add_argument('--predict', type=Path,
                           help='Make prediction on single image')
    eval_group.add_argument('--model', type=Path,
                           help='Model path for prediction')

    # System arguments
    system_group = parser.add_argument_group('⚙️ System Configuration')
    system_group.add_argument('--gpu-memory-limit', type=int,
                             help='GPU memory limit in MB')
    system_group.add_argument('--num-workers', type=int, default=4,
                             help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    # Update configuration with arguments
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.INITIAL_LR = args.learning_rate
    Config.DATA_DIR = args.data_dir
    Config.MODEL_DIR = args.output_dir
    Config.BACKBONE = args.backbone
    Config.USE_ATTENTION = not args.no_attention
    Config.USE_MIXED_PRECISION = not args.no_mixed_precision
    Config.IMAGE_SIZE = tuple(args.image_size)
    Config.OMP_THREADS = str(args.num_workers)

    try:
        if args.predict:
            # Prediction mode
            if not args.model:
                DisplayManager.print_error("Model path required for prediction (use --model)")
                return 1

            DisplayManager.print_section("SINGLE IMAGE PREDICTION", "🔮")

            # Load class names
            class_names = np.load(Config.MODEL_DIR / 'class_names.npy')

            # Make prediction
            result = predict_single_image(args.model, args.predict, class_names)

            if result:
                DisplayManager.print_success(f"Predicted class: {result['predicted_class']}")
                DisplayManager.print_info(f"Confidence: {result['confidence']:.4f}")
                DisplayManager.print_info("Top 3 predictions:")
                for i, (class_name, confidence) in enumerate(result['top_3'], 1):
                    DisplayManager.print_info(f"  {i}. {class_name}: {confidence:.4f}")

        elif args.evaluate:
            # Evaluation mode
            if not args.test_data:
                DisplayManager.print_error("Test data directory required for evaluation (use --test-data)")
                return 1
            evaluate_model(args.evaluate, args.test_data)

        else:
            # Training mode
            DisplayManager.print_section("CONFIGURATION SUMMARY", "📋")
            DisplayManager.print_info(f"Training epochs: {Config.EPOCHS}")
            DisplayManager.print_info(f"Batch size: {Config.BATCH_SIZE}")
            DisplayManager.print_info(f"Learning rate: {Config.INITIAL_LR}")
            DisplayManager.print_info(f"Backbone: {Config.BACKBONE}")
            DisplayManager.print_info(f"Image size: {Config.IMAGE_SIZE}")
            DisplayManager.print_info(f"Attention: {'Enabled' if Config.USE_ATTENTION else 'Disabled'}")
            DisplayManager.print_info(f"Mixed precision: {'Enabled' if Config.USE_MIXED_PRECISION else 'Disabled'}")
            DisplayManager.print_info(f"Fine-tuning: {'Enabled' if args.fine_tune else 'Disabled'}")

            # Start training
            model, history = train_advanced_model()

            if model is None:
                return 1

            # Optional fine-tuning
            if args.fine_tune and model is not None:
                # Recreate datasets for fine-tuning (simplified)
                monitor = AdvancedSystemMonitor()
                pipeline = AdvancedDataPipeline(monitor)
                image_paths, class_names = pipeline.discover_dataset(Config.DATA_DIR)
                labels = [path.parent.name for path in image_paths]

                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
                )

                train_dataset = pipeline.create_tf_dataset(train_paths, train_labels, is_training=True)
                val_dataset = pipeline.create_tf_dataset(val_paths, val_labels, is_training=False)

                steps_per_epoch = len(train_paths) // Config.BATCH_SIZE
                validation_steps = len(val_paths) // Config.BATCH_SIZE

                model = fine_tune_model(model, train_dataset, val_dataset, steps_per_epoch, validation_steps)

        return 0

    except KeyboardInterrupt:
        DisplayManager.print_warning("Process interrupted by user")
        return 1
    except Exception as e:
        DisplayManager.print_error(f"Process failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
