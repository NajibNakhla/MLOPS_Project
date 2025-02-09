import argparse
import os
from model_pipeline import run_pipeline  # Import the function from model_pipeline.py

def main():
    parser = argparse.ArgumentParser(description="Main entry point for MLOps pipeline")
    parser.add_argument('--train', action='store_true', help="Train a machine learning model")
    parser.add_argument('--model', type=str, choices=['decision_tree', 'random_forest'], 
                        required=True, help="Specify the model to train (decision_tree, random_forest)")
    parser.add_argument('--test', action='store_true', help="Run tests for the project")
    args = parser.parse_args()

    if args.train:
        print(f"Training {args.model} model...")
        run_pipeline(args.model)  # Call the run_pipeline function from model_pipeline.py
    
    if args.test:
        print("Running unit tests...")
        os.system('pytest tests/')  # Or another test command depending on your testing framework

if __name__ == '__main__':
    main()
