import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

def plot_feature_distribution(feature_data, feature_name):
    plt.figure(figsize=(10, 6))
    plt.hist(feature_data, bins=30, alpha=0.7, color='blue')
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

class ImprovedDataPreprocessor:
    """
    Enhanced data preprocessing with normalization and constant feature removal
    """
    
    def __init__(self, remove_constant_features=True, feature_threshold=1e-6):
        self.scaler = StandardScaler()
        self.remove_constant_features = remove_constant_features
        self.feature_threshold = feature_threshold
        self.constant_features = []
        self.feature_columns = []
        self.is_fitted = False
        
    def identify_constant_features(self, df):
        """Identify features with zero or near-zero variance"""
        constant_features = []
        feature_stats = {}
        
        for col in df.columns:
            if col == 'height_above':  # Skip target column
                continue
                
            col_data = df[col].dropna()
            if len(col_data) == 0:
                constant_features.append(col)
                feature_stats[col] = {'std': 0, 'unique_values': 0, 'reason': 'no_data'}
                continue
                
            std_val = col_data.std()
            unique_vals = col_data.nunique()
            
            feature_stats[col] = {
                'std': std_val,
                'unique_values': unique_vals,
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean()
            }
            
            # Mark as constant if std is below threshold or only 1 unique value
            if std_val < self.feature_threshold or unique_vals <= 1:
                constant_features.append(col)
                feature_stats[col]['reason'] = 'low_variance' if std_val < self.feature_threshold else 'single_value'
        
        return constant_features, feature_stats
    
    def fit(self, df, target_column='height_above'):
        """Fit the preprocessor on training data"""
        # Identify constant features
        if self.remove_constant_features:
            self.constant_features, feature_stats = self.identify_constant_features(df)
        
        # Prepare feature columns (exclude target and constant features)
        self.feature_columns = [col for col in df.columns 
                               if col != target_column and col not in self.constant_features]
        
        # Fit scaler on non-constant features
        feature_data = df[self.feature_columns]
        self.scaler.fit(feature_data)
        
        self.is_fitted = True
        return self
    
    def transform(self, df, target_column='height_above'):
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Create copy
        df_processed = df.copy()
        
        # Remove constant features
        if self.constant_features:
            df_processed = df_processed.drop(columns=self.constant_features)
        
        # Apply scaling to feature columns
        feature_data = df_processed[self.feature_columns]
        scaled_features = self.scaler.transform(feature_data)
        
        # Replace original features with scaled versions
        df_processed[self.feature_columns] = scaled_features
        
        return df_processed
    
    def fit_transform(self, df, target_column='height_above'):
        """Fit and transform in one step"""
        return self.fit(df, target_column).transform(df, target_column)

class ImprovedRouting_Classifier(nn.Module):
    """
    Improved routing classifier using MLP architectures instead of LSTM/Conv1D
    Better suited for tabular data with proper normalization
    """
    
    def __init__(self, routed_data, hyperparams=None):
        super(ImprovedRouting_Classifier, self).__init__()
        
        self.routed_data = routed_data
        
        # Default hyperparameters for MLP architecture
        default_hyperparams = {
            'mlp_hidden_size': 64,
            'mlp_num_layers': 3,
            'dropout_rate': 0.3,
            'use_residual': True,
            'use_batch_norm': False
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
        self.hyperparams = default_hyperparams
        
        # Get route dimensions
        route1_dim = routed_data['Route1'].shape[1] if 'Route1' in routed_data else 0
        route2_dim = routed_data['Route2'].shape[1] if 'Route2' in routed_data else 0
        route3_dim = routed_data['Route3'].shape[1] if 'Route3' in routed_data else 0
        
        # MLP hyperparameters
        hidden_size = self.hyperparams['mlp_hidden_size']
        num_layers = self.hyperparams['mlp_num_layers']
        dropout_rate = self.hyperparams['dropout_rate']
        use_batch_norm = self.hyperparams['use_batch_norm']
        
        # Route 1: Geography (simple MLP for lat/lon)
        if route1_dim > 0:
            self.route1_mlp = self._create_mlp(route1_dim, hidden_size // 2, num_layers - 1, dropout_rate, use_batch_norm)
            route1_output_dim = hidden_size // 2
        else:
            self.route1_mlp = None
            route1_output_dim = 0
        
        # Route 2: DEM/Wind/Rain/Time (larger MLP for heterogeneous features)
        if route2_dim > 0:
            self.route2_mlp = self._create_mlp(route2_dim, hidden_size, num_layers, dropout_rate, use_batch_norm)
            route2_output_dim = hidden_size
        else:
            self.route2_mlp = None
            route2_output_dim = 0
        
        # Route 3: Sentinel/Soil/Land/Weather (larger MLP for complex features)
        if route3_dim > 0:
            self.route3_mlp = self._create_mlp(route3_dim, hidden_size, num_layers, dropout_rate, use_batch_norm)
            route3_output_dim = hidden_size
        else:
            self.route3_mlp = None
            route3_output_dim = 0
        
        # Fusion layer
        total_features = route1_output_dim + route2_output_dim + route3_output_dim
        
        if total_features == 0:
            raise ValueError("No valid routes found!")
        
        # Final prediction layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_features, hidden_size),
            nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Optional: Direct connection for residual learning
        if self.hyperparams['use_residual']:
            self.direct_connection = nn.Linear(total_features, 1)
        else:
            self.direct_connection = None
    
    def _create_mlp(self, input_dim, hidden_dim, num_layers, dropout_rate, use_batch_norm):
        """Create a multi-layer perceptron"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through all routes"""
        route_outputs = []
        
        # Process Route 1 (Geography)
        if self.route1_mlp is not None and 'Route1' in x:
            route1_out = self.route1_mlp(x['Route1'])
            route_outputs.append(route1_out)
        
        # Process Route 2 (DEM/Wind/Rain/Time)
        if self.route2_mlp is not None and 'Route2' in x:
            route2_out = self.route2_mlp(x['Route2'])
            route_outputs.append(route2_out)
        
        # Process Route 3 (Sentinel/Soil/Land/Weather)
        if self.route3_mlp is not None and 'Route3' in x:
            route3_out = self.route3_mlp(x['Route3'])
            route_outputs.append(route3_out)
        
        if not route_outputs:
            raise ValueError("No route outputs generated!")
        
        # Concatenate route outputs
        fused_features = torch.cat(route_outputs, dim=1)
        
        # Main prediction
        main_output = self.fusion_layers(fused_features)
        
        # Optional residual connection
        if self.direct_connection is not None:
            residual_output = self.direct_connection(fused_features)
            output = main_output + 0.1 * residual_output  # Small residual weight
        else:
            output = main_output
        
        return output.squeeze(-1)  # Remove last dimension for MSE loss

# Legacy compatibility - keep original class for backward compatibility
class Routing_Classifier(ImprovedRouting_Classifier):
    """
    Legacy compatibility wrapper - uses improved architecture by default
    """
    
    def __init__(self, routed_data, hyperparams=None):
        # Convert old hyperparameters to new format if needed
        if hyperparams:
            new_hyperparams = {}
            
            # Map old LSTM parameters to MLP parameters
            if 'lstm_hidden_size' in hyperparams:
                new_hyperparams['mlp_hidden_size'] = hyperparams['lstm_hidden_size']
            if 'lstm_num_layers' in hyperparams:
                new_hyperparams['mlp_num_layers'] = hyperparams['lstm_num_layers']
            
            # Keep existing parameters
            for key in ['dropout_rate', 'mlp_hidden_size', 'mlp_num_layers', 'use_residual', 'use_batch_norm']:
                if key in hyperparams:
                    new_hyperparams[key] = hyperparams[key]
            
            hyperparams = new_hyperparams
        
        super().__init__(routed_data, hyperparams)
