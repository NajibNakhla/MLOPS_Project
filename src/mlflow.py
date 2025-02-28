import subprocess
import time

def start_mlflow():
    """Start MLflow server as a background process."""
    print("🚀 Starting MLflow server...")
    mlflow_process = subprocess.Popen(
        ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(5)  # Wait for the server to initialize
    return mlflow_process
