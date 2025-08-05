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
    
    for data, targets in tqdm.tqdm(train_loader, desc="Training", disable=not Verbose):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader, desc="Evaluating"):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets.float())
            
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            total_loss += loss.item()
    
    accuracy = correct / total
    return total_loss / len(test_loader), accuracy


def train_model(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001, device='cpu', optimizer_type='Adam'):
    """
    Complete training pipeline with configurable hyperparameters
    """
    # Choose optimizer
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.BCELoss()
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
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
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            if Verbose:
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def create_dummy_data_loader(batch_size=32, num_samples=1000):
    """
    Create dummy data loader for testing
    """
    features = torch.randn(num_samples, 10)
    labels = torch.randint(0, 2, (num_samples,)).float()
    
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
