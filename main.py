import argparse
import os
from model_pipeline import run_pipeline  # Import training pipeline
from fastapi_app.api import start_fastapi  # Import FastAPI function
from src.mlflow import start_mlflow

def main():
    parser = argparse.ArgumentParser(description="Main entry point for MLOps pipeline")
    parser.add_argument('--train', action='store_true', help="Train a machine learning model")
    parser.add_argument('--model', type=str, choices=['decision_tree', 'random_forest','logistic_regression','NN'], 
                        help="Specify the model to train (decision_tree, random_forest,logistic_regression, NN)")
    parser.add_argument('--test', action='store_true', help="Run tests for the project")
    parser.add_argument('--api', action='store_true', help="Start FastAPI server")

    args = parser.parse_args()

    if args.train:
        if not args.model:
            print("❌ Please specify a model to train using --model")
            return

        print("🚀 Starting MLflow for training...")
        mlflow_process = start_mlflow() 
        print(f"🎯 Training {args.model} model...")
        run_pipeline(args.model)  

    if args.test:
        print("🧪 Running unit tests...")
        os.system('pytest tests/')  

    if args.api:
        start_fastapi()  # Start FastAPI

if __name__ == '__main__':
    main()
