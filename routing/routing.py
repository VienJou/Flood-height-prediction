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

FEATURE_ROUTE = {
    'DEM': 'Route1',
    'Imagery': 'Route1',
    'Rainfall': 'Route2',
    'Soil': 'Route2',
    'Landuse': 'Route3',
    'Vegetation': 'Route3',
    'Impervious_Surface': 'Route3'
}

def plot_feature_distribution(feature_data, feature_name):
    plt.figure(figsize=(10, 6))
    plt.hist(feature_data, bins=30, alpha=0.7, color='blue')
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# def prepare_routed_data():
#     routed_data = {}
#     for path in FEATURE_ROUTE.values():
#         routed_data[path] = []
    
#     for feature_name, path in FEATURE_ROUTE.items():
#         feature_data = pd.read_csv(f'{path}/{feature_name}.csv')
#         routed_data[path].append(feature_data)
#         plot_feature_distribution(feature_data[feature_name], feature_name)

#     return routed_data


# def prepare_routed_data_with_mapping(feature_route_map, use_dummy_data=True):
#     """
#     Prepare routed data with custom feature-route mapping
#     """
#     routed_data = {'Route1': [], 'Route2': [], 'Route3': []}
    
#     if use_dummy_data:
#         # Create dummy data for testing
#         feature_sizes = {
#             'DEM': 100, 'Imagery': 200, 'Rainfall': 150, 'Soil': 120,
#             'Landuse': 300, 'Vegetation': 250, 'Impervious_Surface': 180
#         }
        
#         for feature_name, route in feature_route_map.items():
#             size = feature_sizes.get(feature_name, 100)
#             feature_data = pd.DataFrame(np.random.randn(100, size))
#             routed_data[route].append(feature_data)
#     else:
#         # Use real data (original implementation)
#         for feature_name, route in feature_route_map.items():
#             feature_data = pd.read_csv(f'{route}/{feature_name}.csv')
#             routed_data[route].append(feature_data)
#     return routed_data

class Routing_Classifier(nn.Module):
    def __init__(self, routed_data, hyperparams=None):
        super(Routing_Classifier, self).__init__()
        self.data = routed_data
        
        # Default hyperparameters
        default_hyperparams = {
            'lstm_hidden_size': 20,
            'lstm_num_layers': 2,
            'dropout_rate': 0.2,
            'conv_channels': (16, 32),
            'fc_hidden_size': 128
        }
        
        # Update with provided hyperparameters
        if hyperparams:
            default_hyperparams.update(hyperparams)
        self.hyperparams = default_hyperparams

        # Calculate input sizes based on actual data
        route1_size = self._get_route_input_size('Route1')
        route2_size = self._get_route_input_size('Route2')
        route3_size = self._get_route_input_size('Route3')

        # Route 1 - LSTM pathway
        self.p1_lstm = nn.LSTM(
            input_size=route1_size, 
            hidden_size=self.hyperparams['lstm_hidden_size'], 
            num_layers=self.hyperparams['lstm_num_layers'], 
            batch_first=True
        )
        self.p1_dropout = nn.Dropout(self.hyperparams['dropout_rate'])
        self.p1_batchnorm = nn.BatchNorm1d(self.hyperparams['lstm_hidden_size'])
        self.p1_fc = nn.Linear(self.hyperparams['lstm_hidden_size'], 1)

        # Route 2 - Dense pathway
        self.p2_fc = nn.Linear(route2_size, 1)
        self.p2_batchnorm = nn.BatchNorm1d(route2_size) if route2_size > 1 else None

        # Route 3 - CNN pathway
        route3_channels = 3
        self.route3_channels = route3_channels
        
        self.p3_conv1 = nn.Conv2d(
            in_channels=route3_channels, 
            out_channels=self.hyperparams['conv_channels'][0], 
            kernel_size=3, 
            padding=1
        )
        self.p3_pool = nn.MaxPool2d(kernel_size=2, stride=2)                            
        self.p3_conv2 = nn.Conv2d(
            self.hyperparams['conv_channels'][0], 
            self.hyperparams['conv_channels'][1], 
            3, 
            padding=1
        )                                  
        self.p3_fc1 = nn.Linear(
            self.hyperparams['conv_channels'][1] * 8 * 8, 
            self.hyperparams['fc_hidden_size']
        )                                           
        self.p3_fc2 = nn.Linear(self.hyperparams['fc_hidden_size'], 1)
        
        # Fusion layer - MultiheadAttention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=1,
            num_heads=1,
            batch_first=True
        )
        self.fusion_fc = nn.Linear(3, 1)

        for unique_path in set(['Route1', 'Route2', 'Route3']):
            if unique_path in self.data and self.data[unique_path]:
                self.data[unique_path] = pd.concat(self.data[unique_path], ignore_index=True)
            else:
                # Create dummy data if route is empty
                self.data[unique_path] = pd.DataFrame(np.random.randn(10, max(1, route2_size if unique_path == 'Route2' else 10)))

    def _get_route_input_size(self, route_name):
        if route_name in self.data and self.data[route_name]:
            total_features = 0
            for df in self.data[route_name]:
                total_features += df.shape[1]
            return total_features
        else:
            defaults = {'Route1': 10, 'Route2': 42, 'Route3': 3072}
            return defaults.get(route_name, 10)

    def forward(self, x=None): # x is unused, but good practice for nn.Module
        # Route 1 - LSTM
        p1_data = self.data['Route1']
        # Assuming p1_data is already a batch of sequences
        p1_tensor = torch.tensor(p1_data.values, dtype=torch.float32).to(self.p1_fc.weight.device)
        if len(p1_tensor.shape) == 2:
            p1_tensor = p1_tensor.unsqueeze(0) # Add batch dimension if not present
        
        p1_lstm_out, _ = self.p1_lstm(p1_tensor)
        p1_out = self.p1_fc(self.p1_batchnorm(self.p1_dropout(p1_lstm_out[:, -1, :])))
        p1_out = torch.sigmoid(p1_out)

        # Route 2 - Dense
        p2_data = self.data['Route2']
        p2_tensor = torch.tensor(p2_data.values, dtype=torch.float32).to(self.p2_fc.weight.device)
        if len(p2_tensor.shape) == 1:
            p2_tensor = p2_tensor.unsqueeze(0)
            
        if self.p2_batchnorm is not None:
            p2_tensor = self.p2_batchnorm(p2_tensor)
        p2_out = self.p2_fc(p2_tensor)
        p2_out = torch.sigmoid(p2_out)

        # Route 3 - CNN
        p3_data = self.data['Route3']
        p3_tensor = torch.tensor(p3_data.values, dtype=torch.float32).to(self.p3_conv1.weight.device)
        
        batch_size = p3_tensor.size(0)
        expected_features = self.route3_channels * 32 * 32
        
        if p3_tensor.size(1) != expected_features:
            if p3_tensor.size(1) < expected_features:
                padding = torch.zeros(batch_size, expected_features - p3_tensor.size(1), device=p3_tensor.device)
                p3_tensor = torch.cat([p3_tensor, padding], dim=1)
            else:
                p3_tensor = p3_tensor[:, :expected_features]
        
        p3_tensor = p3_tensor.view(batch_size, self.route3_channels, 32, 32)
        p3_conv1_out = F.relu(self.p3_conv1(p3_tensor))
        p3_pool_out = self.p3_pool(p3_conv1_out)
        p3_conv2_out = F.relu(self.p3_conv2(p3_pool_out))
        p3_pool_out = self.p3_pool(p3_conv2_out)
        p3_flat = p3_pool_out.view(p3_pool_out.size(0), -1)
        p3_out = self.p3_fc2(F.relu(self.p3_fc1(p3_flat)))
        p3_out = torch.sigmoid(p3_out)

        # Fusion with MultiheadAttention
        fusion_input = torch.stack([p1_out, p2_out, p3_out], dim=1)
        
        attn_output, _ = self.fusion_attention(fusion_input, fusion_input, fusion_input)
        
        fusion_out = self.fusion_fc(attn_output.view(batch_size, -1))
        fusion_out = torch.sigmoid(fusion_out)

        return fusion_out.squeeze(-1)