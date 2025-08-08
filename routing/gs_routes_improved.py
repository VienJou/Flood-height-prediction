from itertools import product
import json
import time
import os
import random
import math
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from routing_improved import ImprovedRouting_Classifier, ImprovedDataPreprocessor
from training_improved import train_model, create_improved_data_loader, set_seed

VERBOSE = False

# Define feature column mappings based on combined_features.csv
FEATURE_COLUMN_MAPPING = {
    'Geography': ['latitude', 'longitude'],
    'DEM': ['dem_min', 'dem_max', 'dem_mean', 'dem_iqr'],
    'Sentinel': ['VV_Min', 'VV_Max', 'VV_Mean', 'VV_IQR', 'VV_SD', 
                 'VH_Min', 'VH_Max', 'VH_Mean', 'VH_IQR', 'VH_SD', 'VH_VV_Ratio'],
    'Soil_Moisture': ['soil_moisture'],
    'Land_Cover': [
        'total_area_km2', 'pct_area_1', 'pct_area_2', 'area_km_1', 'area_km_2', 'cai_1', 'cai_2',
        'pct_area_forest', 'area_km_forest', 'cai_forest', 
        'pct_area_shrubland', 'area_km_shrubland', 'cai_shrubland', 
        'pct_area_herbaceous', 'area_km_herbaceous', 'cai_herbaceous', 
        'pct_area_planted', 'area_km_planted', 'cai_planted', 
        'pct_area_wetlands', 'area_km_wetlands', 'cai_wetlands'
    ],
    'Wind': ['maxwind_kph_day', 'wind_kph', 'wind_degree', 'gust_kph'],
    'Rain': ['precip_mm', 'precipitation'],
    'Other_Weather': [
        'pressure_mb', 'humidity', 'dewpoint_c', 'uv', 'feelslike_c', 
        'windchill_c', 'heatindex_c', 'chance_of_rain', 'chance_of_snow', 'vis_km'
    ],
    'Time': ['year', 'month_sin', 'day_sin', 'hour_sin']
}

# Improved hyperparameter grid for MLP architecture
HYPERPARAMETER_GRID = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'batch_size': [32, 64, 128],
    'epochs': [100],
    'mlp_hidden_size': [64, 128, 256],        # MLP hidden size
    'mlp_num_layers': [2, 3, 4],              # Number of MLP layers
    'dropout_rate': [0.1, 0.2, 0.3],
    'use_residual': [True, False],            # Residual connections
    'use_batch_norm': [True, False],          # Batch normalization
    'optimizer_type': ['AdamW', 'Adam']
}

# Quick test grid for 100 runs
QUICK_HYPERPARAMETER_GRID = {
    'learning_rate': [1e-3, 5e-4],
    'batch_size': [64, 128],
    'epochs': [50],
    'mlp_hidden_size': [128, 256],
    'mlp_num_layers': [3],
    'dropout_rate': [0.2],
    'use_residual': [True],
    'use_batch_norm': [True, False],
    'optimizer_type': ['AdamW']
}

def load_combined_dataset(data_path='../data/combined_features.csv'):
    """
    Load the unified combined_features.csv dataset
    """
    try:
        df = pd.read_csv(data_path)
        # Drop the unnamed column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        if VERBOSE:
            print(f"Loaded dataset with shape: {df.shape}")
            print(f"Available columns: {list(df.columns)}")
            
            # Handle missing values
            print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing values with column means for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def generate_all_feature_route_combinations():
    """
    Generate all possible combinations of feature-to-route assignments
    """
    features = list(FEATURE_COLUMN_MAPPING.keys())
    routes = ['Route1', 'Route2', 'Route3']

    route_assignments = product(routes, repeat=len(features))
    combinations = []
    
    for assignment in route_assignments:
        feature_mapping = dict(zip(features, assignment))
        
        # Create column mapping for this combination
        column_mapping = {}
        for feature, route in feature_mapping.items():
            column_mapping[feature] = {
                'route': route,
                'columns': FEATURE_COLUMN_MAPPING[feature]
            }
        
        combinations.append({
            'feature_mapping': feature_mapping,
            'column_mapping': column_mapping
        })
    
    return combinations

def prepare_routed_data_with_mapping(mapping_config, df, preprocessor=None):
    """
    Prepare routed data with improved preprocessing
    """
    if preprocessor is None:
        # Create and fit preprocessor
        preprocessor = ImprovedDataPreprocessor(remove_constant_features=True)
        df_processed = preprocessor.fit_transform(df, target_column='height_above')
    else:
        df_processed = preprocessor.transform(df, target_column='height_above')
    
    column_mapping = mapping_config['column_mapping']
    
    # Group columns by route
    route_data = {}
    route_column_counts = {}
    
    for route in ['Route1', 'Route2', 'Route3']:
        route_columns = []
        
        for feature_group, config in column_mapping.items():
            if config['route'] == route:
                # Only include columns that exist in processed data and aren't constant
                available_columns = [col for col in config['columns'] 
                                   if col in df_processed.columns and col in preprocessor.feature_columns]
                route_columns.extend(available_columns)
        
        if route_columns:
            route_data[route] = df_processed[route_columns].values
            route_column_counts[route] = len(route_columns)
        else:
            # Create minimal placeholder if no features
            route_data[route] = np.zeros((len(df_processed), 1))
            route_column_counts[route] = 1
    
    return route_data

def improved_grid_search_experiment(
    unified_df, 
    feature_mapping_config, 
    hyperparams, 
    combination_seed, 
    experiment_id=0,
    use_quick_grid=False
):
    """
    Run a single improved grid search experiment
    """
    try:
        # Set seed for reproducibility
        set_seed(combination_seed)
        
        # Create and fit preprocessor
        preprocessor = ImprovedDataPreprocessor(remove_constant_features=True)
        preprocessor.fit(unified_df, target_column='height_above')
        
        # Prepare routed data
        routed_data = prepare_routed_data_with_mapping(feature_mapping_config, unified_df, preprocessor)
        target_values = unified_df['height_above'].values
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ImprovedRouting_Classifier(routed_data, hyperparams)
        model.to(device)
        
        # Create data loaders
        train_loader, val_loader = create_improved_data_loader(
            routed_data, target_values,
            batch_size=hyperparams['batch_size'],
            test_split=0.2,
            random_state=combination_seed
        )
        
        # Train model
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=hyperparams['epochs'],
            learning_rate=hyperparams['learning_rate'],
            device=device,
            optimizer_type=hyperparams['optimizer_type'],
            seed=combination_seed
        )
        
        # Extract best metrics (convert to Python native types)
        best_val_loss = float(min(results['val_losses'])) if results['val_losses'] else float('inf')
        best_val_rmse = float(min(results['val_rmse_scores'])) if results['val_rmse_scores'] else float('inf')
        best_val_mae = float(min(results['val_mae_scores'])) if results['val_mae_scores'] else float('inf')
        
        final_val_loss = float(results['val_losses'][-1]) if results['val_losses'] else float('inf')
        final_val_rmse = float(results['val_rmse_scores'][-1]) if results['val_rmse_scores'] else float('inf')
        final_val_mae = float(results['val_mae_scores'][-1]) if results['val_mae_scores'] else float('inf')
        
        return {
            'experiment_id': experiment_id,
            'combination_seed': combination_seed,
            'feature_mapping': feature_mapping_config['feature_mapping'],
            'hyperparams': hyperparams,
            'best_val_loss': best_val_loss,
            'best_val_rmse': best_val_rmse,
            'best_val_mae': best_val_mae,
            'final_val_loss': final_val_loss,
            'final_val_rmse': final_val_rmse,
            'final_val_mae': final_val_mae,
            'train_losses': [float(x) for x in results['train_losses']],
            'val_losses': [float(x) for x in results['val_losses']],
            'val_rmse_scores': [float(x) for x in results['val_rmse_scores']],
            'val_mae_scores': [float(x) for x in results['val_mae_scores']],
            'success': True,
            'error': None,
            'trained_model': model  # Store the actual model object temporarily
        }
        
    except Exception as e:
        return {
            'experiment_id': experiment_id,
            'combination_seed': combination_seed,
            'feature_mapping': feature_mapping_config.get('feature_mapping', {}),
            'hyperparams': hyperparams,
            'best_val_loss': float('inf'),
            'best_val_rmse': float('inf'),
            'best_val_mae': float('inf'),
            'final_val_loss': float('inf'),
            'final_val_rmse': float('inf'),
            'final_val_mae': float('inf'),
            'success': False,
            'error': str(e)
        }

def run_improved_grid_search(
    max_combinations=100,
    data_path='../data/combined_features.csv',
    save_path='improved_grid_search_results.json',
    checkpoint_interval=10,
    use_quick_grid=True
):
    """
    Run improved grid search with preprocessing and MLP architecture
    """
    print(f"🚀 Starting improved grid search (max {max_combinations} combinations)")
    print(f"Using {'QUICK' if use_quick_grid else 'FULL'} hyperparameter grid")
    
    # Load dataset
    unified_df = load_combined_dataset(data_path)
    if unified_df is None:
        print("❌ Failed to load dataset!")
        return None
    
    print(f"✅ Dataset loaded: {unified_df.shape}")
    
    # Generate feature combinations
    feature_combinations = generate_all_feature_route_combinations()
    print(f"📋 Generated {len(feature_combinations)} feature routing combinations")
    
    # Choose hyperparameter grid
    hyperparam_grid = QUICK_HYPERPARAMETER_GRID if use_quick_grid else HYPERPARAMETER_GRID
    
    # Generate hyperparameter combinations
    hyperparam_keys = list(hyperparam_grid.keys())
    hyperparam_values = list(hyperparam_grid.values())
    hyperparam_combinations = list(product(*hyperparam_values))
    
    print(f"⚙️  Generated {len(hyperparam_combinations)} hyperparameter combinations")
    print(f"🎯 Total possible combinations: {len(feature_combinations) * len(hyperparam_combinations)}")
    
    # Limit combinations
    total_combinations = min(max_combinations, len(feature_combinations) * len(hyperparam_combinations))
    print(f"🔬 Running {total_combinations} experiments")
    
    # Results storage
    results = []
    best_result = {'best_val_loss': float('inf')}
    
    experiment_id = 0
    start_time = time.time()
    
    for feature_idx, feature_config in enumerate(feature_combinations):
        if experiment_id >= max_combinations:
            break
            
        for hyperparam_idx, hyperparam_values in enumerate(hyperparam_combinations):
            if experiment_id >= max_combinations:
                break
            
            # Create hyperparameter dictionary
            hyperparams = dict(zip(hyperparam_keys, hyperparam_values))
            
            # Create combination seed
            combination_seed = 42 + experiment_id
            
            print(f"\n🔬 Experiment {experiment_id + 1}/{total_combinations}")
            print(f"   Feature mapping: {feature_config['feature_mapping']}")
            print(f"   Hyperparams: {hyperparams}")
            print(f"   Seed: {combination_seed}")
            
            # Run experiment
            result = improved_grid_search_experiment(
                unified_df=unified_df,
                feature_mapping_config=feature_config,
                hyperparams=hyperparams,
                combination_seed=combination_seed,
                experiment_id=experiment_id,
                use_quick_grid=use_quick_grid
            )
            
            results.append(result)
            
            # Track best result
            if result['success'] and result['best_val_loss'] < best_result['best_val_loss']:
                best_result = result
                print(f"   🎯 NEW BEST: Val Loss = {result['best_val_loss']:.6f}")
                
                # Save the best model
                if 'trained_model' in result and result['trained_model'] is not None:
                    model_save_path = save_path.replace('.json', '_best_model.pth')
                    torch.save({
                        'model_state_dict': result['trained_model'].state_dict(),
                        'feature_mapping': result['feature_mapping'],
                        'hyperparams': result['hyperparams'],
                        'best_val_loss': result['best_val_loss'],
                        'experiment_id': result['experiment_id'],
                        'combination_seed': result['combination_seed']
                    }, model_save_path)
                    print(f"   💾 Best model saved: {model_save_path}")
                    
                    # Remove model object before JSON serialization
                    result.pop('trained_model', None)
            else:
                print(f"   📊 Val Loss = {result['best_val_loss']:.6f}")
                # Remove model object before JSON serialization
                result.pop('trained_model', None)
            
            # Save checkpoint
            if (experiment_id + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'results': results,
                    'best_result': best_result,
                    'experiment_count': experiment_id + 1,
                    'total_combinations': total_combinations,
                    'timestamp': datetime.now().isoformat()
                }
                
                checkpoint_path = save_path.replace('.json', f'_checkpoint_{experiment_id + 1}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
            
            experiment_id += 1
    
    # Final results
    elapsed_time = time.time() - start_time
    
    final_data = {
        'results': results,
        'best_result': best_result,
        'experiment_count': len(results),
        'total_time_seconds': elapsed_time,
        'average_time_per_experiment': elapsed_time / len(results) if results else 0,
        'hyperparameter_grid': hyperparam_grid,
        'feature_combinations_tested': len(set(r['experiment_id'] for r in results)),
        'successful_experiments': len([r for r in results if r['success']]),
        'failed_experiments': len([r for r in results if not r['success']]),
        'timestamp': datetime.now().isoformat(),
        'use_quick_grid': use_quick_grid
    }
    
    # Save final results
    with open(save_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"\n🎉 IMPROVED GRID SEARCH COMPLETE!")
    print(f"   Total experiments: {len(results)}")
    print(f"   Successful: {final_data['successful_experiments']}")
    print(f"   Failed: {final_data['failed_experiments']}")
    print(f"   Total time: {elapsed_time:.1f}s")
    print(f"   Average time per experiment: {final_data['average_time_per_experiment']:.1f}s")
    print(f"   Best validation loss: {best_result['best_val_loss']:.6f}")
    print(f"   Results saved to: {save_path}")
    
    return final_data

# Legacy compatibility functions
def prepare_routed_data(route_assignments, unified_df):
    """Legacy compatibility wrapper"""
    # Convert old format to new format
    feature_mapping = {}
    for feature, route in route_assignments.items():
        feature_mapping[feature] = route
    
    mapping_config = {
        'feature_mapping': feature_mapping,
        'column_mapping': {}
    }
    
    for feature, route in feature_mapping.items():
        mapping_config['column_mapping'][feature] = {
            'route': route,
            'columns': FEATURE_COLUMN_MAPPING.get(feature, [])
        }
    
    return prepare_routed_data_with_mapping(mapping_config, unified_df)

def create_data_loader(routed_data, target_column='height_above', batch_size=32, test_split=0.2, random_state=42):
    """Legacy compatibility wrapper"""
    # Extract target values from unified_df (this is a limitation of the legacy interface)
    target_values = np.random.randn(routed_data['Route1'].shape[0])  # Placeholder
    return create_improved_data_loader(routed_data, target_values, batch_size, test_split, random_state)

def create_dummy_data_loader(route_sizes, n_samples=100, batch_size=32):
    """Create dummy data loader for testing"""
    routed_data = {}
    for route, size in route_sizes.items():
        routed_data[route] = np.random.randn(n_samples, size)
    
    target_values = np.random.randn(n_samples)
    return create_improved_data_loader(routed_data, target_values, batch_size, test_split=0.2)
