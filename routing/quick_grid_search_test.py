#!/usr/bin/env python3
"""
Quick grid search test with improved architecture

This script runs a 100-experiment grid search to test the improved routing model
with proper feature normalization, constant feature removal, and MLP architecture.
"""

import os
import sys
import time
import json
from datetime import datetime

# Import improved modules
from gs_routes_improved import run_improved_grid_search, set_seed

def main():
    """
    Run quick 100-experiment grid search test
    """
    print("🚀 QUICK GRID SEARCH TEST - IMPROVED ARCHITECTURE")
    print("=" * 60)
    print("Improvements included:")
    print("✅ Feature normalization with StandardScaler")
    print("✅ Constant feature removal")
    print("✅ Simplified MLP architecture")
    print("✅ Better data preprocessing")
    print("✅ Improved training loop")
    print("=" * 60)
    
    # Set global seed for reproducibility
    set_seed(42)
    
    # Configuration
    MAX_COMBINATIONS = 100
    DATA_PATH = '../data/combined_features.csv'
    SAVE_PATH = f'quick_grid_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    CHECKPOINT_INTERVAL = 10
    
    print(f"📋 Configuration:")
    print(f"   Max combinations: {MAX_COMBINATIONS}")
    print(f"   Data path: {DATA_PATH}")
    print(f"   Results path: {SAVE_PATH}")
    print(f"   Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print()
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("   Please make sure the combined_features.csv file exists")
        return
    
    # Run grid search
    start_time = time.time()
    
    try:
        results = run_improved_grid_search(
            max_combinations=MAX_COMBINATIONS,
            data_path=DATA_PATH,
            save_path=SAVE_PATH,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            use_quick_grid=True  # Use quick hyperparameter grid
        )
        
        if results is None:
            print("❌ Grid search failed!")
            return
        
        # Print summary
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 QUICK GRID SEARCH SUMMARY:")
        print(f"   Total experiments: {results['experiment_count']}")
        print(f"   Successful experiments: {results['successful_experiments']}")
        print(f"   Failed experiments: {results['failed_experiments']}")
        print(f"   Success rate: {100 * results['successful_experiments'] / results['experiment_count']:.1f}%")
        print()
        
        print(f"⏱️  TIMING:")
        print(f"   Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"   Average per experiment: {results['average_time_per_experiment']:.1f} seconds")
        print()
        
        print(f"🎯 BEST RESULT:")
        best = results['best_result']
        print(f"   Best validation loss: {best['best_val_loss']:.6f}")
        print(f"   Best validation RMSE: {best['best_val_rmse']:.6f}")
        print(f"   Best validation MAE: {best['best_val_mae']:.6f}")
        print(f"   Feature mapping: {best['feature_mapping']}")
        print(f"   Best hyperparams:")
        for key, value in best['hyperparams'].items():
            print(f"     {key}: {value}")
        print()
        
        # Compare with Random Forest baseline
        rf_baseline = 0.25
        print(f"📈 COMPARISON WITH BASELINES:")
        print(f"   Random Forest baseline: ~{rf_baseline:.3f}")
        if best['best_val_loss'] < rf_baseline:
            improvement = ((rf_baseline - best['best_val_loss']) / rf_baseline * 100)
            print(f"   🎯 SUCCESS: Beat Random Forest by {improvement:.1f}%!")
        else:
            deficit = ((best['best_val_loss'] - rf_baseline) / rf_baseline * 100)
            print(f"   ⚠️  Above Random Forest baseline by {deficit:.1f}%")
        
        # Previous best from original architecture
        original_best = 0.308202
        print(f"   Original architecture best: {original_best:.6f}")
        if best['best_val_loss'] < original_best:
            improvement = ((original_best - best['best_val_loss']) / original_best * 100)
            print(f"   ✅ Improved over original by {improvement:.1f}%!")
        else:
            deficit = ((best['best_val_loss'] - original_best) / original_best * 100)
            print(f"   📊 Change from original: +{deficit:.1f}%")
        
        print(f"\n💾 Results saved to: {SAVE_PATH}")
        
        # Print top 5 results
        successful_results = [r for r in results['results'] if r['success']]
        if len(successful_results) >= 5:
            top_5 = sorted(successful_results, key=lambda x: x['best_val_loss'])[:5]
            print(f"\n🏆 TOP 5 RESULTS:")
            for i, result in enumerate(top_5, 1):
                print(f"   {i}. Val Loss: {result['best_val_loss']:.6f}")
                print(f"      Feature mapping: {result['feature_mapping']}")
                print(f"      Hidden size: {result['hyperparams']['mlp_hidden_size']}, "
                      f"Layers: {result['hyperparams']['mlp_num_layers']}, "
                      f"LR: {result['hyperparams']['learning_rate']}")
                print()
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Grid search interrupted by user")
        print(f"   Partial results may be saved in checkpoint files")
    
    except Exception as e:
        print(f"\n❌ Grid search failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
