import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Verbose = False

def set_seed(seed=42):
    """
    Set seed for reproducibility across all random number generators
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

def collate_batch(batch):
    """Custom collate function for improved data loading"""
    if not batch:
        return None
    
    # Get all route keys from first sample
    route_keys = [key for key in batch[0].keys() if key != 'target']
    
    # Batch each route separately
    batched = {}
    for route in route_keys:
        route_data = [sample[route] for sample in batch]
        batched[route] = torch.stack(route_data)
    
    # Batch targets
    targets = torch.stack([sample['target'] for sample in batch])
    
    return batched, targets

def create_improved_data_loader(routed_data, target_values, batch_size=32, test_split=0.2, random_state=42):
    """
    Create improved data loaders with proper handling for the new architecture
    """
    # Create dataset
    n_samples = len(target_values)
    indices = np.arange(n_samples)
    
    # Split indices
    train_idx, val_idx = train_test_split(
        indices, test_size=test_split, random_state=random_state, 
        stratify=None  # Remove stratification for regression
    )
    
    # Create datasets
    def create_dataset(indices):
        dataset = []
        for idx in indices:
            sample = {}
            for route, data in routed_data.items():
                sample[route] = torch.FloatTensor(data[idx])
            sample['target'] = torch.FloatTensor([target_values[idx]])
            dataset.append(sample)
        return dataset
    
    train_dataset = create_dataset(train_idx)
    val_dataset = create_dataset(val_idx)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    
    return train_loader, val_loader

def train_improved_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device='cpu', seed=42, optimizer_type='AdamW'):
    """
    Improved training loop with better monitoring and techniques
    """
    set_seed(seed)
    
    model.to(device)
    
    # Choose optimizer
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    criterion = torch.nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_rmse_scores = []
    val_mae_scores = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        for batch_data, targets in train_loader:
            # Move to device
            for route in batch_data:
                batch_data[route] = batch_data[route].to(device)
            targets = targets.squeeze(-1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            train_samples += targets.size(0)
        
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_samples = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, targets in val_loader:
                # Move to device
                for route in batch_data:
                    batch_data[route] = batch_data[route].to(device)
                targets = targets.squeeze(-1).to(device)
                
                # Forward pass
                outputs = model(batch_data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * targets.size(0)
                val_samples += targets.size(0)
                
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        # Calculate additional metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
        val_mae = np.mean(np.abs(val_predictions - val_targets))
        
        val_rmse_scores.append(val_rmse)
        val_mae_scores.append(val_mae)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rmse_scores': val_rmse_scores,
        'val_mae_scores': val_mae_scores,
        'best_val_loss': best_val_loss,
        'final_predictions': val_predictions.tolist(),
        'final_targets': val_targets.tolist()
    }
    
    return results

# Legacy compatibility functions
def train_hybrid_model(model, train_loader, optimizer, criterion, device='cpu'):
    """
    Legacy compatibility for hybrid model training
    """
    model.train()
    total_loss = 0.0
    
    for batch_data, targets in tqdm.tqdm(train_loader, desc="Training", disable=not Verbose):
        # Handle both old and new data formats
        if isinstance(batch_data, dict):
            route_data = batch_data
        else:
            # Old format: separate tensors
            r1_data, r2_data, r3_data = batch_data[:3] if len(batch_data) >= 3 else (batch_data[0], batch_data[0], batch_data[0])
            route_data = {
                'Route1': r1_data.to(device),
                'Route2': r2_data.to(device),
                'Route3': r3_data.to(device)
            }
        
        # Move route data to device
        for route in route_data:
            route_data[route] = route_data[route].to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(route_data)
        
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def train(model, train_loader, optimizer, criterion, device='cpu'):
    """
    Standard training function - delegates appropriately
    """
    return train_hybrid_model(model, train_loader, optimizer, criterion, device)

def evaluate(model, test_loader, criterion, device='cpu'):
    """
    Evaluation function with improved handling
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_data, targets in tqdm.tqdm(test_loader, desc="Evaluating", disable=not Verbose):
            # Handle both old and new data formats
            if isinstance(batch_data, dict):
                route_data = batch_data
            else:
                # Old format: separate tensors
                r1_data, r2_data, r3_data = batch_data[:3] if len(batch_data) >= 3 else (batch_data[0], batch_data[0], batch_data[0])
                route_data = {
                    'Route1': r1_data.to(device),
                    'Route2': r2_data.to(device),
                    'Route3': r3_data.to(device)
                }
            
            # Move route data to device
            for route in route_data:
                route_data[route] = route_data[route].to(device)
            targets = targets.to(device)
            
            outputs = model(route_data)
            loss = criterion(outputs, targets.float())
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def evaluate_with_metrics(model, test_loader, device='cpu'):
    """
    Enhanced evaluation function that calculates MSE, RMSE, and MAE
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, targets in tqdm.tqdm(test_loader, desc="Evaluating", disable=not Verbose):
            # Handle both old and new data formats
            if isinstance(batch_data, dict):
                route_data = batch_data
            else:
                # Old format: separate tensors
                r1_data, r2_data, r3_data = batch_data[:3] if len(batch_data) >= 3 else (batch_data[0], batch_data[0], batch_data[0])
                route_data = {
                    'Route1': r1_data.to(device),
                    'Route2': r2_data.to(device),
                    'Route3': r3_data.to(device)
                }
            
            # Move route data to device
            for route in route_data:
                route_data[route] = route_data[route].to(device)
            targets = targets.to(device)
            
            outputs = model(route_data)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def train_model(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001, device='cpu', optimizer_type='Adam', seed=42):
    """
    Complete training pipeline for regression with improved handling
    """
    # Use improved training if possible
    if hasattr(model, 'route1_mlp') and val_loader is not None:
        return train_improved_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
            optimizer_type=optimizer_type
        )
    
    # Fallback to legacy training
    set_seed(seed)
    
    # Choose optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    val_rmse_scores = []
    val_mae_scores = []
    
    for epoch in range(epochs):
        # Training
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validation
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            # Calculate additional metrics
            val_metrics = evaluate_with_metrics(model, val_loader, device)
            val_rmse_scores.append(val_metrics['rmse'])
            val_mae_scores.append(val_metrics['mae'])
        
        if Verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}", end="")
            if val_loader:
                print(f", Val Loss = {val_loss:.6f}")
            else:
                print()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rmse_scores': val_rmse_scores,
        'val_mae_scores': val_mae_scores
    }

# Legacy data loader function for backward compatibility
def create_data_loader(routed_data, target_column='height_above', batch_size=32, test_split=0.2, random_state=42):
    """
    Legacy data loader function - now uses improved version
    """
    # Extract target values
    if isinstance(routed_data, dict) and 'Route1' in routed_data:
        # New format with numpy arrays
        n_samples = routed_data['Route1'].shape[0]
        # Target should be passed separately in new architecture
        target_values = np.zeros(n_samples)  # Placeholder
    else:
        # Try to extract from dataframe format
        target_values = routed_data.get(target_column, np.zeros(100)).values if hasattr(routed_data.get(target_column, None), 'values') else np.zeros(100)
    
    return create_improved_data_loader(routed_data, target_values, batch_size, test_split, random_state)

def create_dummy_data_loader(route_sizes, n_samples=100, batch_size=32):
    """
    Create dummy data loader for testing
    """
    # Create dummy routed data
    routed_data = {}
    for i, (route, size) in enumerate(route_sizes.items()):
        routed_data[route] = np.random.randn(n_samples, size)
    
    # Create dummy targets
    target_values = np.random.randn(n_samples)
    
    return create_improved_data_loader(routed_data, target_values, batch_size, test_split=0.2)
