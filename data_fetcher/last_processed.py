import json
import logging

def load_last_processed_id():
    try:
        with open('last_processed_id.json', 'r') as file:
            data = json.load(file)
            return data.get('last_processed_id', 0)
    except FileNotFoundError:
        logging.warning("last_processed_id.json not found, starting from ID 0.")
        return 0
    except Exception as e:
        logging.error(f"Error loading last processed ID: {e}")
        return 0

def save_last_processed_id(last_processed_id):
    try:
        with open('last_processed_id.json', 'w') as file:
            json.dump({'last_processed_id': last_processed_id}, file)
    except Exception as e:
        logging.error(f"Error saving last processed ID: {e}")