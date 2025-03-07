from elasticsearch import Elasticsearch
import psutil
import json
from datetime import datetime

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")  # Ensure this is correct

def send_log_to_elasticsearch(log_message, cpu_percent, memory_used, memory_total, disk_used, disk_total):
    doc = {
        "model_name": "system_metrics",  # You can change this if you want to categorize the metrics
        "message": log_message,
        "cpu_percent": cpu_percent,
        "memory_used": memory_used,
        "memory_total": memory_total,
        "disk_used": disk_used,
        "disk_total": disk_total,
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Explicitly set the Content-Type header to application/json
        es.index(index="system_metrics", body=json.dumps(doc), headers={"Content-Type": "application/json"})
        print("✅ System metrics sent to Elasticsearch!")
    except Exception as e:
        print(f"❌ Error sending log to Elasticsearch: {e}")

# Function to get system metrics
def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    # Prepare log message (could be customized based on the desired message)
    log_message = f"CPU: {cpu_percent}%, Memory Used: {memory.used / (1024 ** 2):.2f} MB, Disk Used: {disk.used / (1024 ** 3):.2f} GB"

    # Send data to Elasticsearch
    send_log_to_elasticsearch(log_message, cpu_percent, memory.used, memory.total, disk.used, disk.total)

# Collect system metrics and send to Elasticsearch
get_system_metrics()
