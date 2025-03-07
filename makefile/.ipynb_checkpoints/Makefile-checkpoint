SHELL := /bin/bash
VENV_DIR := venv
SRC_DIR := src
IMAGE_NAME := najibnakhla1809/fastapiimage:latest
DOCKERFILE := Dockerfile

# Setup virtual environment
setup:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Start MLflow server
mlflow:
	$(VENV_DIR)/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Train models
train:
	$(VENV_DIR)/bin/python model_pipeline.py --model=$(MODEL)

# Start FastAPI server
run-api:
	$(VENV_DIR)/bin/uvicorn $(SRC_DIR).api:app --host 0.0.0.0 --port 8000 --reload

# Run tests and linting
lint:
	$(VENV_DIR)/bin/flake8 $(SRC_DIR)
	$(VENV_DIR)/bin/black --check $(SRC_DIR)

test: lint
	$(VENV_DIR)/bin/pytest tests/

# Clean virtual environment
clean:
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/**/__pycache__

# Build Docker image
build-docker:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

# Run Docker container locally (to test FastAPI app)
run-docker:
	docker run -d -p 8000:8000 --name fastapi-container $(IMAGE_NAME)

# Push Docker image to Docker Hub
push-docker:
	@docker login  || (echo "Docker login failed, please login manually" && exit 1)
	docker push $(IMAGE_NAME)

# Clean up Docker images (optional)
clean-docker:
	docker rmi $(IMAGE_NAME)
# Pull Docker image from Docker Hub
pull-docker:
	docker pull $(IMAGE_NAME)

# Build, run and test Docker image
build-run-test-docker: build-docker run-docker test

