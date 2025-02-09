pipeline {
    agent any

    environment {
        VENV_DIR = "venv"  // Virtual environment directory
    }

    stages {
        stage('Setup Environment') {
            steps {
                echo 'Creating Virtual Environment & Installing Dependencies...'
                sh 'python3 -m venv $VENV_DIR'
                sh 'source $VENV_DIR/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Code Quality Check') {
            steps {
                echo 'Running Code Quality Checks...'
                sh 'source $VENV_DIR/bin/activate && flake8 src/'  // Linting
                sh 'source $VENV_DIR/bin/activate && black --check src/'  // Formatting check
                sh 'source $VENV_DIR/bin/activate && bandit -r src/'  // Security check
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running Unit Tests...'
                sh 'source $VENV_DIR/bin/activate && pytest tests/'  // Run tests
            }
        }

        stage('Train Decision Tree Model') {
            steps {
                echo 'Training Decision Tree Model...'
                sh 'source $VENV_DIR/bin/activate && python main.py --train --model decision_tree'
            }
        }

        stage('Archive Model') {
            steps {
                echo 'Saving trained model...'
                archiveArtifacts artifacts: 'models/*.pkl', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully! ✅'
        }
        failure {
            echo 'Pipeline failed. Check the logs for details. ❌'
        }
    }
}
