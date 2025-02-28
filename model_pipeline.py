import argparse
import joblib
import mlflow
import mlflow.sklearn
from src.model_process_data import prepare_data
from src.model_balance_data import prepare_model_data
from src.model_train import train_decision_tree,train_random_forest,train_logistic_regression, train_neural_network, train_svm
from src.model_evaluate import evaluate_model

# Set MLflow tracking URI (Make sure MLflow is running)
mlflow.set_tracking_uri("http://localhost:5000")

# File path to dataset
file_path = "data/merged_churn.csv"

def run_pipeline(model_name):
    """End-to-end ML pipeline with model selection and MLflow tracking"""

    # Start MLflow experiment
    mlflow.set_experiment("Churn_Prediction")

    with mlflow.start_run():
        # Log model name as a parameter
        mlflow.log_param("model_name", model_name)

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
        elif model_name == "random_forest":
        	model = train_random_forest(X_train, y_train)
        elif model_name == "logistic_regression":
            model = train_logistic_regression(X_train, y_train)
        elif model_name == "svm":
            model = train_svm(X_train, y_train)
        elif model_name == "NN":
            model = train_neural_network(X_train, y_train, hidden_layer_sizes=(100,), activation="relu", solver="adam", random_state=42)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        #  Evaluate the model
        print(f"\nEvaluating {model_name} model...")
        accuracy, report, cm = evaluate_model(model, X_test, y_test)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)

        # Print results
        print(f"\n--- {model_name} Model Evaluation ---")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", report)
        print("\nConfusion Matrix:\n", cm)

        #  Save the trained model
        print("\nSaving model...")
        model_path = f"models/{model_name}_model.pkl"
        joblib.dump(model, model_path)
        print("Model saved successfully!")

        # Log the trained model in MLflow
        mlflow.sklearn.log_model(model, "model")

        print("\n✅ Model and metrics logged to MLflow!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to train (decision_tree, random_forest)")
    args = parser.parse_args()

    run_pipeline(args.model)
