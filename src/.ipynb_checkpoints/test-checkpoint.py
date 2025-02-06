# Import necessary functions
from model_process_data import prepare_data
from model_balance_data import prepare_model_data
from models import( 
train_decision_tree,
evaluate_model
)
# File path to your test CSV file
file_path = "/home/najib-2/najib_nakhla_4ds5_ml_project/data/merged_churn.csv"  # Adjust if needed

# Prepare data
data, state_map, area_map = prepare_data(file_path)

# Display transformed data
print("Transformed Data:")
print(data.head())

# Display frequency maps
print("\nState Frequency Map:")
print(state_map)

print("\nArea Code Frequency Map:")
print(area_map)


#modeling
X_train_balanced, X_test, y_train_balanced, y_test = prepare_model_data(data, target_column="Churn", test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = train_decision_tree(X_train_balanced, y_train_balanced)

# Evaluate model
accuracy, report, cm = evaluate_model(dt_model, X_test, y_test)

# Print evaluation results
print("\n--- Decision Tree Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)
