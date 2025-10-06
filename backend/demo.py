"""
Demo script to test the trained hybrid exoplanet detection system
"""
from hybrid_pipeline import HybridExoplanetDetector
import numpy as np
from pathlib import Path


def test_on_real_data():
    """Test the hybrid system on real test data"""
    
    print("="*60)
    print("ExoGenAI - Hybrid Exoplanet Detection Demo")
    print("="*60)
    
    # Initialize detector with trained model
    print("\n1. Loading trained model...")
    detector = HybridExoplanetDetector(
        model_path='checkpoints/best_model.pt',
        device='cuda'  # Change to 'cpu' if no GPU
    )
    print("   ✓ Model loaded successfully")
    
    # Load test data
    print("\n2. Loading test samples...")
    data_path = Path('data/preprocessed_dataset.npz')
    
    if not data_path.exists():
        print("   ✗ Error: preprocessed_dataset.npz not found!")
        return
    
    data = np.load(data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"   ✓ Loaded {len(X_test)} test samples")
    
    # Test on multiple samples
    print("\n3. Running detection on sample light curves...")
    print("-" * 60)
    
    class_names = ['False Positive', 'Candidate', 'Exoplanet']
    
    # Test 10 samples
    num_samples = min(10, len(X_test))
    
    for i in range(num_samples):
        sample_flux = X_test[i].flatten()
        true_label = class_names[y_test[i]]
        
        # Run detection
        result = detector.detect(sample_flux)
        
        print(f"\nSample {i+1}:")
        print(f"  True Label: {true_label}")
        print(f"  Prediction: {result['prediction'].replace('_', ' ').title()}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Probabilities:")
        print(f"    False Positive: {result['probabilities']['false_positive']:.4f}")
        print(f"    Candidate: {result['probabilities']['candidate']:.4f}")
        print(f"    Exoplanet: {result['probabilities']['exoplanet']:.4f}")
        
        # Show which stage processed it
        if result['stage_2_ml']['used']:
            print(f"  Pipeline: Rule Filter → ML Model")
            print(f"  ML Processing Time: {result['stage_2_ml']['processing_time_ms']:.2f}ms")
        else:
            print(f"  Pipeline: Rule Filter (rejected)")
            print(f"  Reason: {result['stage_1_filter']['reason']}")
        
        print(f"  Total Time: {result['processing_time_ms']:.2f}ms")
        
        # Check if correct
        predicted_class = result['prediction']
        true_class = class_names[y_test[i]].lower().replace(' ', '_')
        is_correct = predicted_class == true_class
        print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    # Overall statistics
    print("\n" + "="*60)
    print("4. Pipeline Statistics")
    print("="*60)
    
    stats = detector.get_stats()
    print(f"  Total Processed: {stats['total_processed']}")
    print(f"  Filtered by Rules: {stats['filtered_by_rules']} ({stats['filter_rate_percent']:.1f}%)")
    print(f"  Analyzed by ML: {stats['analyzed_by_ml']}")
    print(f"  Average Time per Sample: {stats['average_time_ms']:.2f}ms")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


def test_on_custom_csv():
    """Test on a custom uploaded CSV file"""
    
    import pandas as pd
    
    print("\n" + "="*60)
    print("Testing on Custom CSV File")
    print("="*60)
    
    # Example: test on one of the Kaggle CSV files
    csv_path = input("\nEnter path to CSV file (or press Enter to skip): ").strip()
    
    if not csv_path:
        print("Skipped custom CSV test")
        return
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return
    
    print(f"\nLoading {csv_path.name}...")
    df = pd.read_csv(csv_path)
    
    # Assume first column is label, rest is flux
    flux = df.iloc[0, 1:].values
    
    print(f"Loaded light curve with {len(flux)} data points")
    
    # Initialize detector
    detector = HybridExoplanetDetector(
        model_path='checkpoints/best_model.pt',
        device='cuda'
    )
    
    # Detect
    result = detector.detect(flux)
    
    print("\nDetection Results:")
    print(f"  Prediction: {result['prediction'].replace('_', ' ').title()}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Probabilities:")
    print(f"    False Positive: {result['probabilities']['false_positive']:.4f}")
    print(f"    Candidate: {result['probabilities']['candidate']:.4f}")
    print(f"    Exoplanet: {result['probabilities']['exoplanet']:.4f}")
    print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")


if __name__ == "__main__":
    # Test on preprocessed test data
    test_on_real_data()
    
    # Optionally test on custom CSV
    # test_on_custom_csv()