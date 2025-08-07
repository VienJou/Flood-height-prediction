
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import umap


RANDOM_STATE=54

def split_dataset(X, y, test_size=0.3, random_state=RANDOM_STATE):
    # First split: 70% train, 30% temp (which will be split into 15% val + 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Second split: Split the 30% temp into 15% val and 15% test (50-50 split of the temp data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    print('Number of training observations:', y_train.shape[0], f'({y_train.shape[0]/len(y)*100:.1f}%)')
    print('Number of validation observations:', y_val.shape[0], f'({y_val.shape[0]/len(y)*100:.1f}%)')
    print('Number of test observations:', y_test.shape[0], f'({y_test.shape[0]/len(y)*100:.1f}%)')
    return X_train, y_train, X_val, y_val, X_test, y_test 

def z_normalize(X_train, X_val, X_test):
    # Z-score standardization based on training set statistics (prevents data leakage)
    X_mean = np.nanmean(X_train, axis=0)
    X_std = np.nanstd(X_train, axis=0)
    
    # Apply z-score standardization: (X - mean) / std
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    return X_train_norm, X_val_norm, X_test_norm


def calculate_regression_metrics(y_true, y_pred, set_name):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    print(f"\n{set_name} Set Metrics:")
    print("-" * 30)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_actual_vs_predicted(y_true, y_pred, ax, title, metrics):
    # Regression Visualization: Actual vs Predicted
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{title} Set\nR² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_residuals(y_true, y_pred, ax, title):
    # Residual plots
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{title} Set - Residual Plot')
    ax.grid(True, alpha=0.3)


def feature_importance_analysis(best_rf, features_df):
    # Feature Importance Analysis
    feature_importance = best_rf.feature_importances_

    # Get feature names from the processed dataframe (after datetime conversion and numeric selection)
    processed_features_df = features_df.drop(['height_above'], axis=1).select_dtypes(include=[np.number])
    feature_names = processed_features_df.columns.tolist()

    # Ensure lengths match - if not, create generic names
    if len(feature_names) != len(feature_importance):
        print("Warning: Feature names and importance lengths don't match. Using generic names.")
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    return feature_importance, importance_df
    
def plot_importances(importance_df):
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)

    # Horizontal bar plot for top 20 features
    top_features = importance_df.head(min(20, len(importance_df)))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()

    # Add importance values on bars
    for i, v in enumerate(top_features['importance']):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center')
    plt.subplot(2, 1, 1)
    plt.tight_layout()
    plt.show()

    return 


def plot_cumulative_importances(importance_df): 
    # Cumulative importance plot
    cumulative_importance = np.cumsum(importance_df['importance'].values)
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% of importance')
    plt.axhline(y=0.9, color='orange', linestyle='--', label='90% of importance')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return 



# utility functions
def merge_features(data_dir, file_names):
    # Read and merge all files on 'ID' with custom suffixes
    merged_df = pd.read_csv(os.path.join(data_dir, file_names[0]))

    if 'file_id' in merged_df.columns:
        merged_df = merged_df.rename(columns={'file_id': 'ID'})
    unnamed_cols = [col for col in merged_df.columns if col.startswith('Unnamed')]
    merged_df = merged_df.drop(columns=unnamed_cols)

    unique_columns = ['ID', 'peak_date', 'S1_Date']

    for i, file in enumerate(file_names[1:], 1):
        df = pd.read_csv(os.path.join(data_dir, file))
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
        df = df.drop(columns=unnamed_cols)
        cols_to_keep = ['ID'] + [col for col in df.columns
                                if col not in merged_df.columns or col not in unique_columns]
        df_filtered = df[cols_to_keep]
        merged_df = pd.merge(merged_df, df_filtered, on='ID')
    combined_features = merged_df.drop(columns={'year','projection', 'nlcd_year'}, errors='ignore')
    return combined_features

def datetime_cyclical_transform(combined_features):
    # Convert to datetime and extract temporal features
    combined_features['peak_date'] = pd.to_datetime(combined_features['peak_date'], format='%m/%d/%Y %I:%M:%S %p')
    # Set ID as index to preserve it
    combined_features = combined_features.set_index('ID')

    # Extract and encode temporal features in one go
    temporal_features = {
        'month': (combined_features['peak_date'].dt.month, 12),
        'day': (combined_features['peak_date'].dt.day, 31),
        'hour': (combined_features['peak_date'].dt.hour, 24)
    }
    # Create cyclical features and add year
    combined_features['year'] = combined_features['peak_date'].dt.year
    for feature, (values, period) in temporal_features.items():
        combined_features[f'{feature}_sin'] = np.sin(2 * np.pi * values / period)

    combined_features = combined_features.drop(columns=['peak_date', 'S1_Date']).select_dtypes(exclude=['datetime64'])
    return combined_features

def input_missing_values(combined_features):
    imputer = KNNImputer(n_neighbors=3)
    df_imputed_array = imputer.fit_transform(combined_features)

    # Convert back to DataFrame to preserve column names
    df_imputed = pd.DataFrame(df_imputed_array,
                            columns=combined_features.columns,
                            index=combined_features.index)
    df_imputed['year'] = df_imputed['year'].astype('Int64')

    return df_imputed


def find_id_like_columns(df: pd.DataFrame) -> list[str]:
    id_like = []
    for c in df.columns:
        cl = c.lower()
        if (
            cl in {"id", "ids", "index", "record_id", "uid", "gid"}
            or cl.startswith(("id_", "idx"))
            or cl.endswith("_id")
            or cl == "unnamed: 0"
            or cl == "height_above"
        ):
            id_like.append(c)
    return id_like