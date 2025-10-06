"""
Use pre-prepared exoplanet datasets from public sources
Much faster than downloading from MAST!
"""
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split


def download_kepler_dataset_from_kaggle():
    """
    Download pre-prepared Kepler dataset from public sources
    
    This uses the famous "Exoplanet Hunting in Deep Space" dataset
    which has ~5000 labeled light curves
    """
    print("Downloading Kepler exoplanet dataset...")
    
    # Note: You'll need to download this manually from Kaggle or use their API
    # https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
    
    print("""
    TO DOWNLOAD THE DATASET:
    
    Option 1 - Manual Download (Recommended):
    1. Go to: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
    2. Download 'exoTrain.csv' and 'exoTest.csv'
    3. Place them in './data/' folder
    
    Option 2 - Kaggle API:
    1. Install: pip install kaggle
    2. Set up API credentials (follow Kaggle docs)
    3. Run: kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data
    """)


def load_kaggle_kepler_dataset(data_dir: str = '../data'):
    """
    Load the Kaggle Kepler dataset
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    data_path = Path(data_dir)
    
    train_file = data_path / 'exoTrain.csv'
    test_file = data_path / 'exoTest.csv'
    
    if not train_file.exists() or not test_file.exists():
        print("Dataset files not found!")
        download_kepler_dataset_from_kaggle()
        return None, None, None, None
    
    print("Loading Kaggle Kepler dataset...")
    
    # Load training data
    train_df = pd.read_csv(train_file)
    X_train = train_df.iloc[:, 1:].values  # All columns except first (label)
    y_train = train_df.iloc[:, 0].values   # First column is label
    
    # Load test data
    test_df = pd.read_csv(test_file)
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values
    
    # Convert labels: 2 -> 1 (exoplanet), 1 -> 0 (not exoplanet)
    y_train = (y_train == 2).astype(int)
    y_test = (y_test == 2).astype(int)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test


def convert_to_3class_labels(X, y, candidate_ratio: float = 0.3):
    """
    Convert binary labels to 3-class labels
    
    Args:
        X: Features
        y: Binary labels (0=not planet, 1=planet)
        candidate_ratio: Ratio of exoplanets to label as candidates
    
    Returns:
        X, y_3class where y_3class has labels:
        0 = false positive
        1 = candidate
        2 = confirmed exoplanet
    """
    y_3class = y.copy()
    
    # Find all exoplanet indices
    exoplanet_indices = np.where(y == 1)[0]
    
    # Randomly select some to be candidates (label 1)
    num_candidates = int(len(exoplanet_indices) * candidate_ratio)
    candidate_indices = np.random.choice(exoplanet_indices, 
                                        size=num_candidates, 
                                        replace=False)
    
    # Update labels
    y_3class[candidate_indices] = 1  # Candidates
    
    # Remaining exoplanets stay as 2
    exoplanet_mask = (y == 1) & (~np.isin(np.arange(len(y)), candidate_indices))
    y_3class[exoplanet_mask] = 2
    
    # False positives stay as 0
    
    print(f"\n3-class distribution:")
    print(f"  False positives: {np.sum(y_3class == 0)}")
    print(f"  Candidates: {np.sum(y_3class == 1)}")
    print(f"  Confirmed exoplanets: {np.sum(y_3class == 2)}")
    
    return X, y_3class


def prepare_dataset_for_training(seq_length: int = 2048):
    """
    Load and prepare dataset for training
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
        All properly shaped and normalized
    """
    # Try to load Kaggle dataset
    X_train, y_train, X_test, y_test = load_kaggle_kepler_dataset()
    
    if X_train is None:
        print("\nERROR: Could not load dataset!")
        print("Please download the dataset first.")
        return None
    
    # Normalize
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # Z-score normalization per sample
    X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / (X_train.std(axis=1, keepdims=True) + 1e-8)
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / (X_test.std(axis=1, keepdims=True) + 1e-8)
    
    # Resample to target sequence length if needed
    if X_train.shape[1] != seq_length:
        print(f"Resampling from {X_train.shape[1]} to {seq_length}...")
        X_train_resampled = np.zeros((X_train.shape[0], seq_length))
        X_test_resampled = np.zeros((X_test.shape[0], seq_length))
        
        for i in range(X_train.shape[0]):
            indices = np.linspace(0, X_train.shape[1]-1, seq_length)
            X_train_resampled[i] = np.interp(indices, np.arange(X_train.shape[1]), X_train[i])
        
        for i in range(X_test.shape[0]):
            indices = np.linspace(0, X_test.shape[1]-1, seq_length)
            X_test_resampled[i] = np.interp(indices, np.arange(X_test.shape[1]), X_test[i])
        
        X_train = X_train_resampled
        X_test = X_test_resampled
    
    # Convert to 3-class labels
    X_train, y_train = convert_to_3class_labels(X_train, y_train)
    X_test, y_test = convert_to_3class_labels(X_test, y_test)
    
    # Create validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Reshape for PyTorch [N, seq_len, 1]
    X_train = X_train.reshape(-1, seq_length, 1)
    X_val = X_val.reshape(-1, seq_length, 1)
    X_test = X_test.reshape(-1, seq_length, 1)
    
    print(f"\n{'='*60}")
    print("Dataset prepared for training!")
    print(f"{'='*60}")
    print(f"Training set: {X_train.shape}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    print(f"Validation set: {X_val.shape}")
    print(f"  Class distribution: {np.bincount(y_val)}")
    print(f"Test set: {X_test.shape}")
    print(f"  Class distribution: {np.bincount(y_test)}")
    
    # Save preprocessed dataset
    save_path = Path('../data/preprocessed_dataset.npz')
    save_path.parent.mkdir(exist_ok=True)
    np.savez(save_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test)
    print(f"\nSaved preprocessed dataset to: {save_path}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_preprocessed_dataset(data_path: str = '../data/preprocessed_dataset.npz'):
    """
    Load preprocessed dataset
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not Path(data_path).exists():
        print(f"Preprocessed dataset not found at {data_path}")
        print("Run prepare_dataset_for_training() first")
        return None
    
    data = np.load(data_path)
    
    return (data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'])


if __name__ == "__main__":
    print("Preparing dataset for training...")
    print("\nNote: This uses the Kaggle Kepler dataset.")
    print("Make sure you've downloaded it first!\n")
    
    result = prepare_dataset_for_training(seq_length=2048)
    
    if result is not None:
        print("\n✓ Dataset ready for training!")
    else:
        print("\n✗ Failed to prepare dataset")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data")