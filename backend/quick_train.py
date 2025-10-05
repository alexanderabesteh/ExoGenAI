"""
Quick training test - just 5 epochs to verify everything works
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from models.cnn_transformer import CNNTransformerClassifier


def quick_train():
    """Quick training test"""
    
    print("="*60)
    print("QUICK TRAINING TEST (5 epochs)")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n✓ Loading dataset...")
    data_path = Path('./data/preprocessed_dataset.npz')
    
    if not data_path.exists():
        print(f"\n✗ ERROR: Dataset not found at {data_path}")
        print("Run: python utils/use_existing_dataset.py")
        return
    
    data = np.load(data_path)
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    
    # Small subset for quick test
    subset_size = 500
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]
    X_val = X_val[:100]
    y_val = y_val[:100]
    
    print(f"\n  Using subset: Train {len(X_train)}, Val {len(X_val)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=32
    )
    
    # Create model
    print("\n✓ Creating model...")
    model = CNNTransformerClassifier(
        seq_length=2048,
        cnn_base_channels=32,  # Smaller for quick test
        transformer_d_model=128,
        transformer_heads=4,
        transformer_layers=2,
        num_classes=3,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")
    
    # Loss and optimizer
    class_counts = np.bincount(y_train.numpy())
    weights = len(y_train) / (len(class_counts) * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(5):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}/5:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")
    
    print("\n" + "="*60)
    print("✓ QUICK TEST COMPLETE!")
    print("="*60)
    print("\nEverything is working! Ready for full training.")
    print("Run: python train.py")
    
    # Test inference
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        sample = X_val[0:1].to(device)
        probs = model.predict_proba(sample)
        print(f"\nSample prediction:")
        print(f"  P(False Positive): {probs[0][0]:.4f}")
        print(f"  P(Candidate): {probs[0][1]:.4f}")
        print(f"  P(Exoplanet): {probs[0][2]:.4f}")
        print(f"  Predicted class: {torch.argmax(probs[0]).item()}")
        print(f"  True label: {y_val[0].item()}")
    
    print("\n✓ Inference working!")


if __name__ == "__main__":
    quick_train()