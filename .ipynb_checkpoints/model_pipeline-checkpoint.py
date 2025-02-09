import argparse
import joblib
from src.model_process_data import prepare_data
from src.model_balance_data import prepare_model_data
from src.model_train import train_decision_tree
from src.model_evaluate import evaluate_model

# File path to dataset
file_path = "data/merged_churn.csv"

def run_pipeline(model_name):
    """End-to-end machine learning pipeline with model selection"""
    
    #  Load & preprocess data
    print("Loading and preprocessing data...")
    data, state_map, area_map = prepare_data(file_path)
    
    #  Prepare data for modeling
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_model_data(data, target_column="Churn")

    #  Train the selected model
    print(f"\nTraining {model_name} model...")
    if model_name == "decision_tree":
        model = train_decision_tree(X_train, y_train)
    
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    #  Evaluate the model
    print(f"\nEvaluating {model_name} model...")
    accuracy, report, cm = evaluate_model(model, X_test, y_test)
    
    # Print results
    print(f"\n--- {model_name} Model Evaluation ---")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    #  Save the trained model
    print("\nSaving model...")
    joblib.dump(model, f"models/{model_name}_model.pkl")
    print("Model saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train (decision_tree, random_forest)")
    args = parser.parse_args()
    
    run_pipeline(args.model)
