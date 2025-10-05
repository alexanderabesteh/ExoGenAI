"""
Hybrid Exoplanet Detection Pipeline
Combines rule-based BLS filtering with CNN-Transformer ML model
"""
import torch
import numpy as np
from pathlib import Path
import time
from typing import Dict, Tuple, Optional
import sys

sys.path.append('.')
from models.cnn_transformer import CNNTransformerClassifier


class SimpleBLSFilter:
    """
    Simplified BLS-based rule filter for quick screening
    This is a fast approximation without full BLS computation
    Tuned for normalized light curves (mean=0, std=1)
    """
    
    def __init__(self, 
                 min_dip_count: int = 2,
                 dip_threshold: float = 1.5):
        self.min_dip_count = min_dip_count
        self.dip_threshold = dip_threshold
    
    def detect_dips(self, flux: np.ndarray, threshold: float = None) -> int:
        """
        Count significant dips in the light curve
        
        Args:
            flux: Normalized flux array
            threshold: Sigma threshold for dip detection (uses self.dip_threshold if None)
        
        Returns:
            Number of significant dips detected
        """
        if threshold is None:
            threshold = self.dip_threshold
            
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        
        # Points significantly below mean
        dips = flux < (mean_flux - threshold * std_flux)
        
        # Count clusters of dips (not individual points)
        dip_count = 0
        in_dip = False
        
        for is_dip in dips:
            if is_dip and not in_dip:
                dip_count += 1
                in_dip = True
            elif not is_dip:
                in_dip = False
        
        return dip_count
    
    def check_depth_range(self, flux: np.ndarray) -> bool:
        """
        Check if signal has reasonable depth characteristics
        For normalized data (mean=0, std=1), check if there are
        significant dips that could be transits
        
        Args:
            flux: Normalized flux array (mean ~ 0, std ~ 1)
        
        Returns:
            True if has transit-like dips
        """
        min_flux = np.min(flux)
        
        # For normalized data, dips should be at least 1-2 sigma below mean
        # Too shallow: likely just noise
        # This filters out completely flat or very noisy curves
        return min_flux < -1.0
    
    def check_variability(self, flux: np.ndarray, max_std: float = 5.0) -> bool:
        """
        Check if overall variability is reasonable
        
        Args:
            flux: Flux array
            max_std: Maximum acceptable standard deviation
        
        Returns:
            True if variability is in acceptable range
        """
        std = np.std(flux)
        return std < max_std
    
    def filter(self, flux: np.ndarray) -> Tuple[bool, Dict]:
        """
        Apply rule-based filter
        
        Args:
            flux: Normalized flux array [seq_len]
        
        Returns:
            (is_candidate, details_dict)
        """
        start_time = time.time()
        
        # Check 1: Variability
        if not self.check_variability(flux):
            return False, {
                'stage': 'rule_filter',
                'reason': 'abnormal_variability',
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        # Check 2: Depth check
        if not self.check_depth_range(flux):
            return False, {
                'stage': 'rule_filter',
                'reason': 'no_significant_dips',
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        # Check 3: Detect multiple dips (suggests periodicity)
        dip_count = self.detect_dips(flux)
        if dip_count < self.min_dip_count:
            return False, {
                'stage': 'rule_filter',
                'reason': 'insufficient_transits',
                'dip_count': dip_count,
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        # Passed all filters!
        return True, {
            'stage': 'rule_filter',
            'reason': 'passed_filters',
            'dip_count': dip_count,
            'processing_time_ms': (time.time() - start_time) * 1000
        }


class HybridExoplanetDetector:
    """
    Complete hybrid pipeline: Rule-based filter + ML model
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize hybrid detector
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Stage 1: Rule-based filter
        self.rule_filter = SimpleBLSFilter()
        
        # Stage 2: ML model
        self.ml_model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'filtered_by_rules': 0,
            'analyzed_by_ml': 0,
            'total_time_ms': 0
        }
    
    def load_model(self, model_path: str):
        """Load trained ML model"""
        print(f"Loading ML model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with same config
        config = checkpoint.get('config', {})
        self.ml_model = CNNTransformerClassifier(
            seq_length=config.get('seq_length', 2048),
            cnn_base_channels=config.get('cnn_channels', 64),
            transformer_d_model=config.get('transformer_dim', 256),
            transformer_heads=config.get('transformer_heads', 8),
            transformer_layers=config.get('transformer_layers', 4),
            num_classes=config.get('num_classes', 3),
            dropout=config.get('dropout', 0.3)
        ).to(self.device)
        
        self.ml_model.load_state_dict(checkpoint['model_state_dict'])
        self.ml_model.eval()
        
        print("✓ Model loaded successfully")
    
    def preprocess_flux(self, flux: np.ndarray, target_length: int = 2048) -> np.ndarray:
        """
        Preprocess flux for model input
        
        Args:
            flux: Raw flux array
            target_length: Target sequence length
        
        Returns:
            Preprocessed flux [target_length, 1]
        """
        # Remove NaNs
        flux = flux[~np.isnan(flux)]
        
        # Normalize
        flux = flux / np.median(flux)
        flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)
        
        # Resample to target length
        if len(flux) != target_length:
            indices = np.linspace(0, len(flux) - 1, target_length)
            flux = np.interp(indices, np.arange(len(flux)), flux)
        
        # Reshape for model
        flux = flux.reshape(-1, 1)
        
        return flux
    
    def detect(self, flux: np.ndarray, skip_ml: bool = False) -> Dict:
        """
        Run complete hybrid detection pipeline
        
        Args:
            flux: Input flux array
            skip_ml: If True, only run rule filter (for testing)
        
        Returns:
            Detection results dictionary
        """
        start_time = time.time()
        self.stats['total_processed'] += 1
        
        # Preprocess
        processed_flux = self.preprocess_flux(flux)
        
        # Stage 1: Rule-based filter
        is_candidate, rule_details = self.rule_filter.filter(processed_flux.flatten())
        
        if not is_candidate:
            # Rejected by rules
            self.stats['filtered_by_rules'] += 1
            total_time = (time.time() - start_time) * 1000
            self.stats['total_time_ms'] += total_time
            
            return {
                'prediction': 'false_positive',
                'confidence': 1.0,
                'probabilities': {
                    'false_positive': 1.0,
                    'candidate': 0.0,
                    'exoplanet': 0.0
                },
                'stage_1_filter': rule_details,
                'stage_2_ml': {'used': False, 'reason': 'filtered_by_rules'},
                'final_decision': 'rejected_by_physics',
                'processing_time_ms': total_time
            }
        
        # Stage 2: ML analysis
        if skip_ml or self.ml_model is None:
            total_time = (time.time() - start_time) * 1000
            return {
                'prediction': 'candidate',
                'confidence': 0.5,
                'probabilities': {
                    'false_positive': 0.5,
                    'candidate': 0.5,
                    'exoplanet': 0.0
                },
                'stage_1_filter': rule_details,
                'stage_2_ml': {'used': False, 'reason': 'no_model_loaded'},
                'final_decision': 'passed_rules_only',
                'processing_time_ms': total_time
            }
        
        self.stats['analyzed_by_ml'] += 1
        
        ml_start = time.time()
        
        # Run ML model
        with torch.no_grad():
            flux_tensor = torch.FloatTensor(processed_flux).unsqueeze(0).to(self.device)
            probs = self.ml_model.predict_proba(flux_tensor)
            probs = probs[0].cpu().numpy()
        
        ml_time = (time.time() - ml_start) * 1000
        
        # Get prediction
        pred_class = int(np.argmax(probs))
        class_names = ['false_positive', 'candidate', 'exoplanet']
        prediction = class_names[pred_class]
        confidence = float(probs[pred_class])
        
        total_time = (time.time() - start_time) * 1000
        self.stats['total_time_ms'] += total_time
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'false_positive': float(probs[0]),
                'candidate': float(probs[1]),
                'exoplanet': float(probs[2])
            },
            'stage_1_filter': rule_details,
            'stage_2_ml': {
                'used': True,
                'processing_time_ms': ml_time,
                'model_output': probs.tolist()
            },
            'final_decision': f'ml_classified_as_{prediction}',
            'processing_time_ms': total_time
        }
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        if self.stats['total_processed'] > 0:
            avg_time = self.stats['total_time_ms'] / self.stats['total_processed']
            filter_rate = (self.stats['filtered_by_rules'] / self.stats['total_processed']) * 100
        else:
            avg_time = 0
            filter_rate = 0
        
        return {
            'total_processed': self.stats['total_processed'],
            'filtered_by_rules': self.stats['filtered_by_rules'],
            'filter_rate_percent': filter_rate,
            'analyzed_by_ml': self.stats['analyzed_by_ml'],
            'average_time_ms': avg_time,
            'total_time_ms': self.stats['total_time_ms']
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_processed': 0,
            'filtered_by_rules': 0,
            'analyzed_by_ml': 0,
            'total_time_ms': 0
        }


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Testing Hybrid Exoplanet Detection Pipeline")
    print("="*60)
    
    # Initialize detector (without model for quick test)
    detector = HybridExoplanetDetector()
    
    print("\n1. Testing with synthetic false positive (noisy flat)...")
    # Generate noisy flat light curve
    flux_fp = np.random.normal(1.0, 0.002, 2048)
    result = detector.detect(flux_fp, skip_ml=True)
    print(f"   Result: {result['prediction']} (confidence: {result['confidence']:.2f})")
    print(f"   Reason: {result['stage_1_filter']['reason']}")
    print(f"   Time: {result['processing_time_ms']:.2f}ms")
    
    print("\n2. Testing with synthetic transit signal...")
    # Generate light curve with transit
    flux_transit = np.ones(2048)
    # Add periodic transits
    for t0 in [500, 1000, 1500]:
        transit_mask = np.abs(np.arange(2048) - t0) < 20
        flux_transit[transit_mask] -= 0.01  # 1% depth
    # Add noise
    flux_transit += np.random.normal(0, 0.001, 2048)
    
    result = detector.detect(flux_transit, skip_ml=True)
    print(f"   Result: {result['prediction']} (confidence: {result['confidence']:.2f})")
    print(f"   Reason: {result['final_decision']}")
    print(f"   Dips detected: {result['stage_1_filter'].get('dip_count', 'N/A')}")
    print(f"   Time: {result['processing_time_ms']:.2f}ms")
    
    print("\n3. Processing statistics:")
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ Hybrid pipeline test complete!")
    print("="*60)
    print("\nTo use with trained model:")
    print("  detector = HybridExoplanetDetector(model_path='checkpoints/best_model.pt')")
    print("  result = detector.detect(your_flux_array)")