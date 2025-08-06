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
from torch.utils.data import DataLoader, TensorDataset

Verbose = False

def train(model, train_loader, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0.0
    
    for r1_data, r2_data, r3_data, targets in tqdm.tqdm(train_loader, desc="Training", disable=not Verbose):
        route_data = {
            'Route1': r1_data.to(device),
            'Route2': r2_data.to(device),
            'Route3': r3_data.to(device)
        }
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(route_data)
        
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for r1_data, r2_data, r3_data, targets in tqdm.tqdm(test_loader, desc="Evaluating", disable=not Verbose):
            route_data = {
                'Route1': r1_data.to(device),
                'Route2': r2_data.to(device),
                'Route3': r3_data.to(device)
            }
            targets = targets.to(device)
            
            outputs = model(route_data)
            loss = criterion(outputs, targets.float())
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def train_model(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001, device='cpu', optimizer_type='Adam'):
    """
    Complete training pipeline for regression.
    """
    # Choose optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(epochs):
        if Verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
        
        # Training
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        if Verbose:
            print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            if Verbose:
                print(f"Val Loss: {val_loss:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


def create_dummy_data_loader(model, batch_size=32, num_samples=1000):
    """
    Create dummy data loader for testing, compatible with the model's expected input.
    """
    # Get input sizes from the model's stored data shapes
    route1_size = model.data['Route1'].shape[1]
    route2_size = model.data['Route2'].shape[1]
    route3_size = model.data['Route3'].shape[1]

    # Create dummy tensors for each route
    r1_features = torch.randn(num_samples, route1_size)
    r2_features = torch.randn(num_samples, route2_size)
    r3_features = torch.randn(num_samples, route3_size)
    
    # Dummy labels for regression
    labels = torch.randn(num_samples)
    
    dataset = TensorDataset(r1_features, r2_features, r3_features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
