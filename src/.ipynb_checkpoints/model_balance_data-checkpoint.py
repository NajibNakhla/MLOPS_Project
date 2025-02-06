# Libraries
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

# Function to split features and target
def split_features_target(data, target_column="Churn"):
    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target
    return X, y

# Function to split the dataset
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to apply SMOTEENN for balancing data
def balance_data(X_train, y_train, random_state=42):
    smote_enn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Final function to prepare data for modeling
def prepare_model_data(data, target_column="Churn", test_size=0.2, random_state=42):
    X, y = split_features_target(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train, random_state)
    return X_train_balanced, X_test, y_train_balanced, y_test