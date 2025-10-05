"""
Hybrid CNN-Transformer model for exoplanet detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class ResidualBlock(nn.Module):
    """Residual block for CNN"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class CNNFeatureExtractor(nn.Module):
    """Multi-scale CNN for extracting transit features"""
    
    def __init__(self, input_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # Multi-scale convolutions to capture different transit features
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(base_channels)
        
        self.conv3 = nn.Conv1d(base_channels, base_channels*2, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(base_channels*2)
        
        self.pool1 = nn.MaxPool1d(2)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(base_channels*2, base_channels*2)
        self.res_block2 = ResidualBlock(base_channels*2, base_channels*4)
        
        self.pool2 = nn.MaxPool1d(2)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 1]
        
        Returns:
            [batch, seq_len//4, channels*4]
        """
        # Transpose for Conv1d: [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Multi-scale feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool1(x)  # Downsample by 2
        x = self.dropout(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = self.pool2(x)  # Downsample by 2 again
        x = self.dropout(x)
        
        # Transpose back: [batch, seq_len//4, channels]
        x = x.transpose(1, 2)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling"""
    
    def __init__(self, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            [batch, seq_len, d_model]
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        return x


class CNNTransformerClassifier(nn.Module):
    """
    Hybrid CNN-Transformer model for exoplanet detection
    3-class classification: [false_positive, candidate, exoplanet]
    """
    
    def __init__(self, 
                 seq_length: int = 2048,
                 cnn_base_channels: int = 64,
                 transformer_d_model: int = 256,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 num_classes: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.seq_length = seq_length
        
        # Stage 1: CNN Feature Extraction
        self.cnn = CNNFeatureExtractor(input_channels=1, 
                                      base_channels=cnn_base_channels)
        
        # After pooling twice, sequence length is seq_length // 4
        # After CNN, we have base_channels * 4 channels
        cnn_out_channels = cnn_base_channels * 4
        
        # Project CNN features to transformer dimension
        self.projection = nn.Linear(cnn_out_channels, transformer_d_model)
        
        # Stage 2: Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=transformer_d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Stage 3: Classification Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(transformer_d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: [batch, seq_len, 1] input flux
            return_attention: If True, return attention weights
        
        Returns:
            logits: [batch, num_classes]
            attention_weights: Optional attention weights
        """
        # Stage 1: CNN Feature Extraction
        cnn_features = self.cnn(x)  # [batch, seq_len//4, cnn_channels]
        
        # Project to transformer dimension
        transformer_input = self.projection(cnn_features)  # [batch, seq_len//4, d_model]
        
        # Stage 2: Transformer Encoding
        transformer_output = self.transformer(transformer_input)  # [batch, seq_len//4, d_model]
        
        # Stage 3: Global pooling and classification
        # Transpose for pooling: [batch, d_model, seq_len//4]
        pooled = self.global_pool(transformer_output.transpose(1, 2))  # [batch, d_model, 1]
        pooled = pooled.squeeze(-1)  # [batch, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        if return_attention:
            # Extract attention from last transformer layer
            # This is a simplified version - full implementation would store attention weights
            return logits, None
        
        return logits
    
    def predict_proba(self, x):
        """
        Get probability predictions
        
        Args:
            x: [batch, seq_len, 1] input
        
        Returns:
            probabilities: [batch, num_classes]
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = CNNTransformerClassifier(
        seq_length=2048,
        cnn_base_channels=64,
        transformer_d_model=256,
        transformer_heads=8,
        transformer_layers=4,
        num_classes=3
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_length = 2048
    
    # Random input
    x = torch.randn(batch_size, seq_length, 1)
    
    # Forward pass
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Get probabilities
    probs = model.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities:\n{probs[0]}")
    print(f"Sum of probabilities: {probs[0].sum()}")