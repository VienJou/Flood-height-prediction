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
    'Vegetation': 'Route3'
}

def plot_feature_distribution(feature_data, feature_name):
    plt.figure(figsize=(10, 6))
    plt.hist(feature_data, bins=30, alpha=0.7, color='blue')
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

class Routing_Classifier(nn.Module):
    def __init__(self, routed_data, hyperparams=None):
        super(Routing_Classifier, self).__init__()
        self.data = routed_data

        # First, concatenate dataframes within each route
        for route_name in ['Route1', 'Route2', 'Route3']:
            if route_name in self.data and self.data[route_name]:
                # Concatenate list of dataframes into a single dataframe
                self.data[route_name] = pd.concat(self.data[route_name], axis=1)
            else:
                # If a route is empty, create a placeholder DataFrame
                self.data[route_name] = pd.DataFrame(np.zeros((10, 1))) # Default shape

        # Default hyperparameters for an all-dense architecture
        default_hyperparams = {
            'p2_hidden_size': 64,
            'p3_hidden_size_1': 128,
            'p3_hidden_size_2': 64,
            'dropout_rate': 0.3
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
        self.hyperparams = default_hyperparams

        # Calculate input sizes for each route from the concatenated data
        route1_size = self.data['Route1'].shape[1]
        route2_size = self.data['Route2'].shape[1]
        route3_size = self.data['Route3'].shape[1]

        # --- Define All-Dense Routes ---

        # Route 1: 1 FC Layer
        self.p1_fc1 = nn.Linear(route1_size, 1)

        # Route 2: 2 FC Layers
        self.p2_fc1 = nn.Linear(route2_size, self.hyperparams['p2_hidden_size'])
        self.p2_fc2 = nn.Linear(self.hyperparams['p2_hidden_size'], 1)

        # Route 3: 3 FC Layers
        self.p3_fc1 = nn.Linear(route3_size, self.hyperparams['p3_hidden_size_1'])
        self.p3_fc2 = nn.Linear(self.hyperparams['p3_hidden_size_1'], self.hyperparams['p3_hidden_size_2'])
        self.p3_fc3 = nn.Linear(self.hyperparams['p3_hidden_size_2'], 1)
        
        self.dropout = nn.Dropout(self.hyperparams['dropout_rate'])

        # Fusion layer - MultiheadAttention (remains unchanged)
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=1,
            num_heads=1,
            batch_first=True
        )
        self.fusion_fc = nn.Linear(3, 1)

    def _get_route_input_size(self, route_name):
        # This function is no longer needed as size is calculated directly in __init__
        # but kept to avoid breaking old calls if any. Returns shape from processed data.
        if route_name in self.data:
            return self.data[route_name].shape[1]
        return 1 # Default size

    def forward(self, x=None):
        # The input `x` is a dictionary of tensors for each route
        # This allows for batch processing during training
        device = self.p1_fc1.weight.device
        
        # Route 1
        p1_tensor = x['Route1'].to(device)
        p1_out = self.p1_fc1(p1_tensor)
        # No activation here, will be fused then sigmoid

        # Route 2
        p2_tensor = x['Route2'].to(device)
        p2_out = F.relu(self.p2_fc1(p2_tensor))
        p2_out = self.dropout(p2_out)
        p2_out = self.p2_fc2(p2_out)

        # Route 3
        p3_tensor = x['Route3'].to(device)
        p3_out = F.relu(self.p3_fc1(p3_tensor))
        p3_out = self.dropout(p3_out)
        p3_out = F.relu(self.p3_fc2(p3_out))
        p3_out = self.dropout(p3_out)
        p3_out = self.p3_fc3(p3_out)

        # Prepare for fusion
        # Ensure outputs are in the shape (batch_size, 1)
        if p1_out.dim() == 1: p1_out = p1_out.unsqueeze(1)
        if p2_out.dim() == 1: p2_out = p2_out.unsqueeze(1)
        if p3_out.dim() == 1: p3_out = p3_out.unsqueeze(1)
        
        # Fusion with MultiheadAttention
        # Input shape for MHA: (batch_size, sequence_length, embedding_dim)
        # Here, sequence_length is 3 (for 3 routes), embedding_dim is 1
        fusion_input = torch.cat([p1_out, p2_out, p3_out], dim=1)
        fusion_input = fusion_input.unsqueeze(2) # Add embedding_dim
        
        # MHA expects (Query, Key, Value)
        attn_output, _ = self.fusion_attention(fusion_input, fusion_input, fusion_input)
        
        # Flatten the output of the attention layer before the final FC layer
        # Shape after attention: (batch_size, 3, 1) -> flatten to (batch_size, 3)
        attn_output_flat = attn_output.squeeze(2)
        
        # For regression, we don't apply a sigmoid activation
        final_output = self.fusion_fc(attn_output_flat)

        return final_output.squeeze(-1)