from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
#Function to train a Decision Tree model
def train_decision_tree(X_train, y_train, random_state=42):
    dt_model = DecisionTreeClassifier(random_state=random_state)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model
    
    
# Function to train a Logistic Regression model
def train_logistic_regression(X_train, y_train, random_state=42):
    lr_model = LogisticRegression(random_state=random_state)
    lr_model.fit(X_train, y_train)
    return lr_model

# Function to train a Neural Network model
def train_neural_network(X_train, y_train, hidden_layer_sizes=(100,), activation="relu", solver="adam", random_state=42):
    nn_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, random_state=random_state, max_iter=500)
    nn_model.fit(X_train, y_train)
    return nn_model
