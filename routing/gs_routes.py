from itertools import product
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from routing import Routing_Classifier
from training import train_model, create_dummy_data_loader

VERBOSE = False

def generate_all_feature_route_combinations():
    features = ['DEM', 'Imagery', 'Rainfall', 'Soil', 'Landuse', 'Vegetation', 'Impervious_Surface']
    routes = ['Route1', 'Route2', 'Route3']

    route_assignments = product(routes, repeat=len(features))
    combinations = []
    for assignment in route_assignments:
        feature_mapping = dict(zip(features, assignment))
        combinations.append(feature_mapping)
    
    print(f"Generated {len(combinations)} combinations")
    return combinations

def prepare_routed_data_with_mapping(feature_route_map, use_dummy_data=True):
    """
    Prepare routed data with custom feature-route mapping
    """
    routed_data = {'Route1': [], 'Route2': [], 'Route3': []}
    
    if use_dummy_data:
        # Create dummy data for testing
        feature_sizes = {
            'DEM': 100, 'Imagery': 200, 'Rainfall': 150, 'Soil': 120,
            'Landuse': 300, 'Vegetation': 250, 'Impervious_Surface': 180
        }
        
        for feature_name, route in feature_route_map.items():
            size = feature_sizes.get(feature_name, 100)
            feature_data = pd.DataFrame(np.random.randn(100, size))
            routed_data[route].append(feature_data)
    else:
        for feature_name, route in feature_route_map.items():
            try:
                feature_data = pd.read_csv(f'{route}/{feature_name}.csv')
                routed_data[route].append(feature_data)
            except FileNotFoundError:
                feature_data = pd.DataFrame(np.random.randn(100, 100))
                routed_data[route].append(feature_data)
    
    return routed_data

def run_hyperparameter_sweep_for_combination(feature_mapping, combination_idx, total_combinations):
    if VERBOSE:
        print(f"\n{'='*80}")
        print(f"COMBINATION {combination_idx + 1}/{total_combinations}")
        print(f"{'='*80}")

    if VERBOSE:
        print("Feature Mapping:")
        for route in ['Route1', 'Route2', 'Route3']:
            features_in_route = [f for f, r in feature_mapping.items() if r == route]
            print(f"  {route}: {features_in_route}")
    
    # Open to much modification
    hyperparameter_grid = {
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'epochs': [5, 10],
        'lstm_hidden_size': [20, 32],
        'lstm_num_layers': [1, 2],
        'dropout_rate': [0.1, 0.2],
        'conv_channels': [(16, 32), (32, 64)],
        'fc_hidden_size': [64, 128],
        'optimizer_type': ['Adam', 'SGD']
    }
    
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    hyperparams_combinations = list(product(*param_values))

    if VERBOSE:
        print(f"Running {len(hyperparams_combinations)} hyperparameter combinations...")

    combination_results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for hp_idx, hp_values in enumerate(hyperparams_combinations):
        try:
            hyperparams = dict(zip(param_names, hp_values))

            if VERBOSE:
                print(f"\n  Hyperparameter set {hp_idx + 1}/{len(hyperparams_combinations)}")
                print(f"  LR={hyperparams['learning_rate']}, BS={hyperparams['batch_size']}, "
                      f"LSTM={hyperparams['lstm_hidden_size']}, OPT={hyperparams['optimizer_type']}")
            
            routed_data = prepare_routed_data_with_mapping(feature_mapping, use_dummy_data=True)
            model_hyperparams = {
                'lstm_hidden_size': hyperparams['lstm_hidden_size'],
                'lstm_num_layers': hyperparams['lstm_num_layers'],
                'dropout_rate': hyperparams['dropout_rate'],
                'conv_channels': hyperparams['conv_channels'],
                'fc_hidden_size': hyperparams['fc_hidden_size']
            }
            
            # Create model with hyperparameters
            model = Routing_Classifier(routed_data, model_hyperparams)
            model.to(device)
            
            train_loader = create_dummy_data_loader(
                batch_size=hyperparams['batch_size'], 
                num_samples=800
            )
            val_loader = create_dummy_data_loader(
                batch_size=hyperparams['batch_size'], 
                num_samples=200
            )
            
            results = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=hyperparams['epochs'],
                learning_rate=hyperparams['learning_rate'],
                device=device,
                optimizer_type=hyperparams['optimizer_type']
            )
            
            final_train_loss = results['train_losses'][-1] if results['train_losses'] else float('inf')
            final_val_acc = results['val_accuracies'][-1] if results['val_accuracies'] else 0.0
            best_val_acc = max(results['val_accuracies']) if results['val_accuracies'] else 0.0
            
            experiment_result = {
                'combination_idx': combination_idx,
                'hyperparameter_idx': hp_idx,
                'feature_mapping': feature_mapping,
                'hyperparameters': hyperparams,
                'final_train_loss': final_train_loss,
                'final_val_acc': final_val_acc,
                'best_val_acc': best_val_acc,
                'train_losses': results['train_losses'],
                'val_accuracies': results['val_accuracies'],
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            print(f"Val Acc: {final_val_acc:.4f}, Best: {best_val_acc:.4f}")
            
        except Exception as e:
            # Handle failed experiments
            experiment_result = {
                'combination_idx': combination_idx,
                'hyperparameter_idx': hp_idx,
                'feature_mapping': feature_mapping,
                'hyperparameters': hyperparams,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            print(f"Failed: {str(e)}")
        combination_results.append(experiment_result)
    return combination_results

def run_comprehensive_grid_search(max_feature_combinations=None, save_every=5, 
    random_search=False, num_random_samples=500,
    results_filename='comprehensive_search_results.json'
):

    print("STARTING COMPREHENSIVE SEARCH")
    print(f"Mode: {'Random Search' if random_search else 'Grid Search'}")
    print("="*80)

    all_feature_combinations = generate_all_feature_route_combinations()
    
    hyperparameter_grid = {
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20],
        'p2_hidden_size': [32, 64, 128],
        'p3_hidden_size_1': [64, 128, 256],
        'p3_hidden_size_2': [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.5],
        'optimizer_type': ['Adam', 'SGD', 'RMSprop']
    }
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    all_hyperparams_combinations = [dict(zip(param_names, v)) for v in product(*param_values)]

    # Create a list of all possible experiments (feature_map, hyperparams)
    all_experiments = list(product(all_feature_combinations, all_hyperparams_combinations))
    total_possible_experiments = len(all_experiments)
    print(f"Total possible experiments in full search space: {total_possible_experiments}")

    # 2. Select Experiments to Run
    if random_search:
        import random
        num_samples = min(num_random_samples, len(all_feature_combinations))
        step = len(all_feature_combinations) / num_samples
        spaced_indices = [int(i * step) for i in range(num_samples)]
        spaced_feature_combos = [all_feature_combinations[i] for i in spaced_indices]
        experiments_to_run = []
        for feature_combo in spaced_feature_combos:
            random_hyperparams = random.choice(all_hyperparams_combinations)
            experiments_to_run.append((feature_combo, random_hyperparams))
        print(f"Running {len(experiments_to_run)} experiments, sampled evenly across feature combinations.")
    elif max_feature_combinations:
        print(f"GRID SEARCH: Limiting to first {max_feature_combinations} feature combinations.")
        limited_feature_combos = all_feature_combinations[:max_feature_combinations]
        experiments_to_run = list(product(limited_feature_combos, all_hyperparams_combinations))
        print(f"Running {len(experiments_to_run)} experiments for the first {max_feature_combinations} feature maps.")
    else:
        print("GRID SEARCH: Running all experiments.")
        experiments_to_run = all_experiments

    # 3. Run Experiments
    all_results = []
    start_time = time.time()
    
    for idx, (feature_mapping, hyperparams) in enumerate(experiments_to_run):
        try:
            # For logging, find the original index of the feature mapping
            combo_idx = all_feature_combinations.index(feature_mapping)
        except ValueError:
            combo_idx = -1 # Should not happen

        if VERBOSE:
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {idx + 1}/{len(experiments_to_run)}")
            print(f"Feature Combination Index: {combo_idx}")
            print(f"Hyperparameters: {hyperparams}")
            print(f"{'='*80}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # Prepare data and model for the current experiment
            routed_data = prepare_routed_data_with_mapping(feature_mapping, use_dummy_data=True)
            model_hyperparams = {k: v for k, v in hyperparams.items() if k not in ['learning_rate', 'batch_size', 'epochs', 'optimizer_type']}
            
            model = Routing_Classifier(routed_data, model_hyperparams)
            model.to(device)
            
            train_loader = create_dummy_data_loader(batch_size=hyperparams['batch_size'], num_samples=800)
            val_loader = create_dummy_data_loader(batch_size=hyperparams['batch_size'], num_samples=200)
            
            # Train the model
            results = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                epochs=hyperparams['epochs'], learning_rate=hyperparams['learning_rate'],
                device=device, optimizer_type=hyperparams['optimizer_type']
            )
            
            # Collate results
            final_train_loss = results['train_losses'][-1] if results['train_losses'] else float('inf')
            final_val_acc = results['val_accuracies'][-1] if results['val_accuracies'] else 0.0
            best_val_acc = max(results['val_accuracies']) if results['val_accuracies'] else 0.0
            
            experiment_result = {
                'experiment_idx': idx, 'combination_idx': combo_idx, 'hyperparameters': hyperparams, 
                'feature_mapping': feature_mapping, 'final_train_loss': final_train_loss, 
                'final_val_acc': final_val_acc, 'best_val_acc': best_val_acc,
                'train_losses': results['train_losses'], 'val_accuracies': results['val_accuracies'],
                'timestamp': datetime.now().isoformat(), 'success': True
            }
            print(f"Success - Val Acc: {final_val_acc:.4f}, Best: {best_val_acc:.4f}")

        except Exception as e:
            experiment_result = {
                'experiment_idx': idx, 'combination_idx': combo_idx, 'hyperparameters': hyperparams, 
                'feature_mapping': feature_mapping, 'error': str(e), 
                'timestamp': datetime.now().isoformat(), 'success': False
            }
            print(f"Failed: {str(e)}")
        
        all_results.append(experiment_result)
        
        # 4. Save Checkpoints
        if (idx + 1) % save_every == 0:
            checkpoint_filename = f"checkpoint_{idx + 1}.json"
            with open(checkpoint_filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved checkpoint: {checkpoint_filename}")
        
        # 5. Log Progress
        elapsed = time.time() - start_time
        avg_time_per_exp = elapsed / (idx + 1)
        remaining_exp = len(experiments_to_run) - (idx + 1)
        eta_seconds = avg_time_per_exp * remaining_exp
        
        if VERBOSE:
            print(f"\nPROGRESS: {idx + 1}/{len(experiments_to_run)} | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"ETA: {eta_seconds/60:.1f}m")
    
    # 6. Save Final Results
    with open(results_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"SEARCH COMPLETE")
    print(f"Total experiments run: {len(all_results)}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to: {results_filename}")
    print(f"{'='*80}")
    
    return analyze_comprehensive_results(all_results)

def analyze_comprehensive_results(all_results):
    """
    Analyze results across all combinations and hyperparameters
    """
    print(f"\nCOMPREHENSIVE ANALYSIS")
    print("="*80)

    successful_results = [r for r in all_results if r.get('success', False)]
    
    print(f"Successful experiments: {len(successful_results)}/{len(all_results)}")
    
    if not successful_results:
        print("No successful experiments!")
        return None
    
    # Sort by best validation accuracy
    successful_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    best_config = successful_results[0]
    print(f"\nBEST OVERALL CONFIGURATION:")
    print(f"Best Validation Accuracy: {best_config['best_val_acc']:.4f}")
    print(f"Feature Mapping: {best_config['feature_mapping']}")
    print(f"Hyperparameters: {best_config['hyperparameters']}")
    
    # Best per combination
    combination_best = {}
    for result in successful_results:
        combo_idx = result['combination_idx']
        if combo_idx not in combination_best or result['best_val_acc'] > combination_best[combo_idx]['best_val_acc']:
            combination_best[combo_idx] = result
    
    print(f"\nTOP 10 FEATURE COMBINATIONS:")
    sorted_combinations = sorted(combination_best.values(), key=lambda x: x['best_val_acc'], reverse=True)
    
    for i, config in enumerate(sorted_combinations[:10]):
        print(f"\n{i+1}. Combination {config['combination_idx'] + 1} - Val Acc: {config['best_val_acc']:.4f}")
        for route in ['Route1', 'Route2', 'Route3']:
            features = [f for f, r in config['feature_mapping'].items() if r == route]
            print(f"   {route}: {features}")
        print(f"   Best hyperparams: LR={config['hyperparameters']['learning_rate']}, "
              f"BS={config['hyperparameters']['batch_size']}")
    
    return {
        'best_overall': best_config,
        'best_per_combination': combination_best,
        'all_successful': successful_results
    }

# Example of how to run the search
if __name__ == "__main__":
    # To run a full grid search on the first 3 feature combinations:
    # run_comprehensive_grid_search(max_feature_combinations=3)

    # To run a random search of 500 experiments from the entire space:
    run_comprehensive_grid_search(
        random_search=True, 
        num_random_samples=500,
        save_every=100,
        results_filename='random_search_results.json'
    )

