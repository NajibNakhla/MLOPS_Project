from elasticsearch import Elasticsearch
import json

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")  # Ensure this is correct

def send_log_to_elasticsearch(log_message,accuracy,model_name):
    doc = {
        "model_name" :model_name,
        "message": log_message,
        "accuracy": accuracy,
        "timestamp": "2025-03-06T14:32:00"
    }
    
    try:
        # Explicitly set the Content-Type header to application/json
        es.index(index="allmodels", body=json.dumps(doc), headers={"Content-Type": "application/json"})
        print("✅ Log sent to Elasticsearch!")
    except Exception as e:
        print(f"❌ Error sending log to Elasticsearch: {e}")
