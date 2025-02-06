# Import necessary functions from the model_pipiline_churn.py file
from model_process_data import prepare_data,import_data

# File path to your test CSV file (ensure it's in the same folder)
file_path = '/home/najib-2/najib_nakhla_4ds5_ml_project/data/merged_churn.csv'  # Adjust the path if needed

# Prepare data using your pipeline functions
data, state_map, area_map = prepare_data(file_path)

# Display transformed data
print("Transformed Data:")
print(data.head())

# Display frequency maps
print("\nState Frequency Map:")
print(state_map)

print("\nArea Code Frequency Map:")
print(area_map)
