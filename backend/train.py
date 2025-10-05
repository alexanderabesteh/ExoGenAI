# put near top of script
import os, sys, torch
print("PYTHON EXE:", sys.executable)
print("PID:", os.getpid())
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda device name:", torch.cuda.get_device_name(0))



"""
Training script for CNN-Transformer exoplanet detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import sys
sys.path.append('.')

# Import model
from models.cnn_transformer import CNNTransformerClassifier


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def compute_class_weights(y_train):
    """
    Compute class weights for imbalanced dataset
    
    Args:
        y_train: Training labels
    
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(y_train)
    total = len(y_train)
    
    # Inverse frequency weighting
    weights = total / (len(class_counts) * class_counts)
    
    print(f"\nClass distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples (weight: {weights[i]:.3f})")
    
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def train_model(config):
    """
    Main training function
    
    Args:
        config: Dictionary with training configuration
    """
    print("="*60)
    print("Starting Training")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    data_path = Path(config['data_path'])
    data = np.load(data_path)
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.LongTensor(data['y_test'])
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=config['batch_size'], 
                             shuffle=True)
    val_loader = DataLoader(val_dataset, 
                           batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, 
                            batch_size=config['batch_size'])
    
    # Initialize model
    print("\nInitializing model...")
    model = CNNTransformerClassifier(
        seq_length=config['seq_length'],
        cnn_base_channels=config['cnn_channels'],
        transformer_d_model=config['transformer_dim'],
        transformer_heads=config['transformer_heads'],
        transformer_layers=config['transformer_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Loss function with class weights
    class_weights = compute_class_weights(y_train.numpy())
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=config['learning_rate'] / 100
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # Training loop
    print("\n" + "="*60)
    print("Training Loop")
    print("="*60)
    
    best_val_f1 = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = Path(config['checkpoint_dir']) / 'best_model.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (F1: {val_f1:.4f})")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                               target_names=['False Positive', 'Candidate', 'Exoplanet']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_loss': float(test_loss),
        'best_val_f1': float(best_val_f1),
        'total_epochs': epoch + 1,
        'history': history,
        'confusion_matrix': cm.tolist(),
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = Path(config['checkpoint_dir']) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return model, history, results


if __name__ == "__main__":
    # Training configuration
    config = {
        # Data
        'data_path': './data/preprocessed_dataset.npz',
        'seq_length': 2048,
        'num_classes': 3,
        
        # Model architecture
        'cnn_channels': 64,
        'transformer_dim': 256,
        'transformer_heads': 8,
        'transformer_layers': 4,
        'dropout': 0.3,
        
        # Training
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'early_stopping_patience': 7,
        
        # Checkpoints
        'checkpoint_dir': './checkpoints'
    }
    
    print("Training Configuration:")
    print(json.dumps(config, indent=2))
    print()
    
    # Train
    model, history, results = train_model(config)
    
    print("\nFinal Results:")
    print(f"Best Validation F1: {results['best_val_f1']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1-Score: {results['test_f1']:.4f}")