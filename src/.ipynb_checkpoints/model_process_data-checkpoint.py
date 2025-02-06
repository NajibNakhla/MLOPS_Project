# Libraries for data manipulation and visualization
import pandas as pd
import numpy as np
# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Function to import data
def import_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())  # Display first few rows
    return df

# Function to transform categorical data
def transform_object_data(data):
    data = data.copy()
    
    # Ensure values are strings and remove leading/trailing spaces
    data['International plan'] = data['International plan'].astype(str).str.strip().str.lower()
    data['Voice mail plan'] = data['Voice mail plan'].astype(str).str.strip().str.lower()
    
    # Map categorical values to numerical
    data['International plan'] = data['International plan'].map({'yes': 1, 'no': 0})
    data['Voice mail plan'] = data['Voice mail plan'].map({'yes': 1, 'no': 0})
    
    # Convert Churn column to integer
    data['Churn'] = data['Churn'].astype(int)
    
    return data  # Return modified DataFrame

# Function for frequency encoding
def frequence_encoding_data(data):
    data = data.copy()
    
    # Create frequency dictionaries
    state_freq_dict = data['State'].value_counts().to_dict()
    area_freq_dict = data['Area code'].value_counts().to_dict()

    # Apply mapping
    data['State'] = data['State'].map(state_freq_dict)
    data['Area code'] = data['Area code'].map(area_freq_dict)

    return data, state_freq_dict, area_freq_dict  # Return mappings for reuse

# Function to scale numerical data
def scale_data(data):
    data = data.copy()
    
    # List of numerical columns to scale
    numerical_columns = [
        'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge',
        'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls'
    ]
    
    # Apply StandardScaler
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data  # Return modified DataFrame

# Function to prepare data
def prepare_data(file_path):
    data = import_data(file_path)
    data = transform_object_data(data)
    data, state_map, area_map = frequence_encoding_data(data)  # Store frequency maps
    data = scale_data(data)
    
    return data, state_map, area_map  # Return processed data and mappings
