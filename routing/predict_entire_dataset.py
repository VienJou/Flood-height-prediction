#!/usr/bin/env python3
"""
Script to make predictions on the entire dataset using the best model from grid search.

This script:
1. Loads the best model saved during grid search
2. Prepares the entire dataset using the same preprocessing
3. Makes predictions on all samples
4. Exports results with ID, target value, and predicted value to CSV
"""

import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import argparse

# Import our modules
from routing_improved import ImprovedRouting_Classifier, ImprovedDataPreprocessor
from gs_routes_improved import FEATURE_COLUMN_MAPPING, prepare_routed_data_with_mapping

def load_best_model(model_path):
    """Load the best model from the saved checkpoint"""
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Model info:")
    print(f"  Experiment ID: {checkpoint['experiment_id']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")
    print(f"  Feature mapping: {checkpoint['feature_mapping']}")
    print(f"  Hyperparameters: {checkpoint['hyperparams']}")
    
    return checkpoint

def create_model_from_checkpoint(checkpoint, routed_data):
    """Create and load the model from checkpoint"""
    model = ImprovedRouting_Classifier(routed_data, checkpoint['hyperparams'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def prepare_data_for_prediction(data_path, feature_mapping, seed=42):
    """Prepare the entire dataset for prediction using the same preprocessing"""
    print(f"Loading data from: {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Drop unnamed columns if they exist
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Store original IDs and target values
    ids = df['ID'].values if 'ID' in df.columns else np.arange(len(df))
    target_values = df['height_above'].values if 'height_above' in df.columns else np.full(len(df), np.nan)
    
    # Fill missing values with column means for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Create mapping config for prepare_routed_data_with_mapping
    mapping_config = {
        'feature_mapping': feature_mapping,
        'column_mapping': {}
    }
    
    for feature, route in feature_mapping.items():
        mapping_config['column_mapping'][feature] = {
            'route': route,
            'columns': FEATURE_COLUMN_MAPPING.get(feature, [])
        }
    
    # Create and fit preprocessor (same as in training)
    preprocessor = ImprovedDataPreprocessor(remove_constant_features=True)
    preprocessor.fit(df, target_column='height_above')
    
    # Prepare routed data
    routed_data = prepare_routed_data_with_mapping(mapping_config, df, preprocessor)
    
    print(f"Preprocessed data - removed {len(preprocessor.constant_features)} constant features")
    print(f"Final feature count per route:")
    for route, data in routed_data.items():
        print(f"  {route}: {data.shape[1]} features")
    
    return routed_data, ids, target_values, preprocessor

def make_predictions(model, routed_data, device='cpu', batch_size=64):
    """Make predictions on the entire dataset"""
    print("Making predictions...")
    
    model.to(device)
    model.eval()
    
    # Convert to tensors
    route_tensors = {}
    for route, data in routed_data.items():
        route_tensors[route] = torch.FloatTensor(data)
    
    n_samples = next(iter(route_tensors.values())).shape[0]
    all_predictions = []
    
    # Process in batches to avoid memory issues
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Create batch
            batch_data = {}
            for route, tensor in route_tensors.items():
                batch_data[route] = tensor[start_idx:end_idx].to(device)
            
            # Make predictions
            predictions = model(batch_data)
            all_predictions.append(predictions.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"Predictions completed for {len(predictions)} samples")
    return predictions.flatten()

def save_predictions(ids, target_values, predictions, output_path):
    """Save predictions to CSV file"""
    print(f"Saving predictions to: {output_path}")
    
    # Calculate prediction errors where target values are available
    errors = np.full_like(predictions, np.nan)
    valid_targets = ~np.isnan(target_values)
    errors[valid_targets] = np.abs(predictions[valid_targets] - target_values[valid_targets])
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'ID': ids,
        'target_value': target_values,
        'predicted_value': predictions,
        'prediction_error': errors
    })
    
    # Add summary statistics
    if np.any(valid_targets):
        mse = np.mean((predictions[valid_targets] - target_values[valid_targets]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(errors[valid_targets])
        
        print(f"\nPrediction Summary (on samples with known targets):")
        print(f"  Samples with targets: {np.sum(valid_targets)}")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Target range: [{np.min(target_values[valid_targets]):.3f}, {np.max(target_values[valid_targets]):.3f}]")
        print(f"  Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Results saved with {len(results_df)} rows")
    
    return results_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Make predictions using the best model from grid search')
    parser.add_argument('--model', default='quick_grid_search_results_20250807_202500_best_model.pth',
                       help='Path to the saved model file')
    parser.add_argument('--data', default='../data/combined_features.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--output', default='predictions_entire_dataset.csv',
                       help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for prediction')
    parser.add_argument('--device', default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🔮 PREDICTION ON ENTIRE DATASET")
    print("=" * 50)
    print(f"Model file: {args.model}")
    print(f"Data file: {args.data}")
    print(f"Output file: {args.output}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)
    
    try:
        # Load the best model
        checkpoint = load_best_model(args.model)
        
        # Prepare the data
        routed_data, ids, target_values, preprocessor = prepare_data_for_prediction(
            args.data, checkpoint['feature_mapping']
        )
        
        # Create and load the model
        model = create_model_from_checkpoint(checkpoint, routed_data)
        
        # Make predictions
        predictions = make_predictions(model, routed_data, device, args.batch_size)
        
        # Save results
        results_df = save_predictions(ids, target_values, predictions, args.output)
        
        print(f"\n✅ Prediction completed successfully!")
        print(f"📊 Results saved to: {args.output}")
        
        # Create a summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_file': args.model,
            'data_file': args.data,
            'output_file': args.output,
            'total_samples': len(predictions),
            'samples_with_targets': int(np.sum(~np.isnan(target_values))),
            'model_info': {
                'experiment_id': checkpoint['experiment_id'],
                'best_val_loss': checkpoint['best_val_loss'],
                'feature_mapping': checkpoint['feature_mapping'],
                'hyperparams': checkpoint['hyperparams']
            }
        }
        
        if np.any(~np.isnan(target_values)):
            valid_targets = ~np.isnan(target_values)
            mse = np.mean((predictions[valid_targets] - target_values[valid_targets]) ** 2)
            summary['performance_on_known_targets'] = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(np.mean(np.abs(predictions[valid_targets] - target_values[valid_targets])))
            }
        
        # Save summary
        summary_path = args.output.replace('.csv', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise

if __name__ == "__main__":
    main()
