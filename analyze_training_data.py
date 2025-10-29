#!/usr/bin/env python3
"""
Analyze the processed training data to verify quality
"""

import json
from collections import Counter
import statistics

def analyze_training_data():
    """Analyze the processed training data"""
    
    # Load data
    with open('training_data/combined_training_data.json', 'r') as f:
        data = json.load(f)
    
    print(f"=== Training Data Analysis ===")
    print(f"Total samples: {len(data)}")
    
    # Signal distribution
    signals = [sample['signal'] for sample in data]
    signal_counts = Counter(signals)
    print(f"\nSignal distribution:")
    for signal, count in signal_counts.items():
        print(f"  {signal}: {count} ({count/len(data)*100:.1f}%)")
    
    # Confidence analysis
    confidences = [sample['confidence'] for sample in data]
    print(f"\nConfidence analysis:")
    print(f"  Range: {min(confidences):.2f} - {max(confidences):.2f}")
    print(f"  Average: {statistics.mean(confidences):.2f}")
    print(f"  Median: {statistics.median(confidences):.2f}")
    
    # Technical indicators analysis
    print(f"\nTechnical indicators analysis:")
    sample = data[0]
    tech_indicators = sample['technical_indicators']
    
    # Check RSI range
    rsi_values = [s['technical_indicators']['rsi'] for s in data]
    print(f"  RSI range: {min(rsi_values):.1f} - {max(rsi_values):.1f}")
    
    # Check MACD range
    macd_values = [s['technical_indicators']['macd'] for s in data]
    print(f"  MACD range: {min(macd_values):.1f} - {max(macd_values):.1f}")
    
    # Check price range
    prices = [s['price_data']['close'] for s in data]
    print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    
    # Check volume range
    volumes = [s['price_data']['volume'] for s in data]
    print(f"  Volume range: {min(volumes):.2f} - {max(volumes):.2f}")
    
    # Sample data quality check
    print(f"\nData quality check:")
    print(f"  All samples have signal: {all('signal' in s for s in data)}")
    print(f"  All samples have confidence: {all('confidence' in s for s in data)}")
    print(f"  All samples have technical_indicators: {all('technical_indicators' in s for s in data)}")
    print(f"  All samples have market_context: {all('market_context' in s for s in data)}")
    
    # Check for missing values
    missing_values = 0
    for sample in data:
        for key, value in sample['technical_indicators'].items():
            if value is None or (isinstance(value, float) and (value != value)):  # Check for NaN
                missing_values += 1
                break
    
    print(f"  Samples with missing technical indicator values: {missing_values}")
    
    # Show a few sample entries
    print(f"\nSample entries:")
    for i in [0, 50, 100]:
        if i < len(data):
            sample = data[i]
            print(f"  Sample {i}:")
            print(f"    Signal: {sample['signal']}")
            print(f"    Confidence: {sample['confidence']:.2f}")
            print(f"    RSI: {sample['technical_indicators']['rsi']:.1f}")
            print(f"    MACD: {sample['technical_indicators']['macd']:.1f}")
            print(f"    Price: ${sample['price_data']['close']:.2f}")
            print(f"    Volume: {sample['price_data']['volume']:.2f}")
            print()

if __name__ == "__main__":
    analyze_training_data()
