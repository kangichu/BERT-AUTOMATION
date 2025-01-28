# main.py

import logging
import os
import sys
import schedule
import time
import numpy as np
from flask import Flask, request, jsonify
from data_fetcher.mysql_fetcher import fetch_data_from_mysql, fetch_new_listings_from_mysql
from data_fetcher.last_processed import load_last_processed_id, save_last_processed_id
from narrative_generator.gpt_narrative import generate_narrative_manually, generate_response_with_gpt
from vectorizer.bert_vectorizer import generate_embeddings, generate_single_embedding
from faiss_handler.faiss_store import store_embeddings_in_faiss, FAISSManager
from data_handler.tensorboard_exporter import create_projector_config
from data_handler.tensorboard_exporter import export_to_tensorboard
from utils.logger import setup_logging
from dotenv import load_dotenv

app = Flask(__name__)

def run_pipeline():
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Step 1: Fetch data from MySQL
        rows = fetch_data_from_mysql()

        # Step 2: Generate narratives using GPT
        # narratives_gpt2 = generate_narratives_with_gpt(rows)
        # narratives_bloom =  generate_narratives_with_bloom(rows)
        # narratives_llama =  generate_narratives_with_llama(rows)
        # hf_token = os.getenv('HUGUNF_FACE_TOKEN','')
        # narratives_gemma =  generate_narratives_with_gemma(rows, hf_token)
        narratives_manually =  generate_narrative_manually(rows)

        # Step 3: Generate BERT embeddings
        vector_data = generate_embeddings(narratives_manually)

        # Step 4: Store embeddings in FAISS
        embeddings = [item[2] for item in vector_data]
        metadata = [f"RefID: {item[0]} | Narrative: {item[1]}" for item in vector_data]
        store_embeddings_in_faiss(vector_data)

        # Step 5: Export embeddings for TensorBoard
        export_to_tensorboard(np.array(embeddings), metadata)

        # Step 6: Call the function to create the projector config
        create_projector_config(log_dir="logs/embedding_logs", embeddings=embeddings, metadata=metadata)

        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

def update_pipeline_with_new_listings(index_path):
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Step 1: Load the last processed ID
        last_processed_id = load_last_processed_id()

        # Step 2: Fetch new listings from MySQL
        new_rows, new_last_processed_id = fetch_new_listings_from_mysql(last_processed_id)

        if not new_rows:
            logging.info("No new listings to process.")
            return

        # Step 3: Generate narratives for new listings
        narratives_manually = generate_narrative_manually(new_rows)

        # Step 4: Generate BERT embeddings for new narratives
        vector_data = generate_embeddings(narratives_manually)

        # Step 5: Load the existing FAISS index
        faiss_manager = FAISSManager()
        faiss_index = faiss_manager.load_index(index_path)

        # Step 6: Add new embeddings to FAISS
        for item in vector_data:
            id, narrative, embedding = item
            faiss_index.add(np.array([embedding]))

        # Step 7: Save the updated FAISS index
        faiss_manager.save_index(faiss_index, index_path)

        # Step 8: Update the last processed ID
        save_last_processed_id(new_last_processed_id)

        # Step 9: Optionally, you might want to export new embeddings for TensorBoard
        # Here, we're not doing this since it might not be necessary for each update
        # new_embeddings = [item[2] for item in vector_data]
        # new_metadata = [f"RefID: {item[0]} | Narrative: {item[1]}" for item in vector_data]
        # export_to_tensorboard(np.array(new_embeddings), new_metadata)

        logging.info(f"Pipeline update completed successfully! Added {len(new_rows)} new listings.")

    except Exception as e:
        logging.critical(f"Pipeline update failed: {e}")

def search_faiss(query, index_path):
    try:
        # Load the FAISS index
        faiss_index = FAISSManager.load_index(index_path)
        
        # Generate embedding for the query
        query_embedding = generate_single_embedding(query)
        
        # Search for the most similar entries
        D, I = faiss_index.search(np.array([query_embedding]), k=5)  # k=5 for top 5 results
        results = [faiss_index.reconstruct(int(i)) for i in I[0]]
        
        # Retrieve the narratives from the results, assuming they are stored in the format (id, narrative, embedding)
        retrieved_narratives = [result[1] for result in results if len(result) > 1]
        
        return retrieved_narratives
    
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

def generate_response(query, retrieved_narratives):
    # Here we use a hypothetical function to generate a response based on the query and retrieved narratives
    # You would need to implement or adjust this function according to your actual model setup
    response = generate_response_with_gpt(query, retrieved_narratives)
    return response

def scheduled_update():
    update_pipeline_with_new_listings('faiss_index')
    logging.info("Scheduled update completed.")

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    user_query = data['query']
    index_path = 'faiss_index'  # Ensure this path is correct
    
    logging.info(f"Received query: {user_query}")
    
    retrieved_narratives = search_faiss(user_query, index_path)
    
    logging.info(f"Retrieved narratives: {retrieved_narratives}")
    
    if retrieved_narratives:
        response = generate_response(user_query, retrieved_narratives)
        logging.info(f"Generated Response: {response}")
        return jsonify({"response": response}), 200
    else:
        logging.info("No relevant results found.")
        return jsonify({"response": "No relevant results found."}), 200

@app.route('/update_pipeline', methods=['POST'])
def update_pipeline_endpoint():
    try:
        update_pipeline_with_new_listings('faiss_index')
        return jsonify({"status": "Update completed successfully"}), 200
    except Exception as e:
        logging.error(f"Error during update: {e}")
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    setup_logging()
    if '--run-pipeline' in sys.argv:
        run_pipeline()

    # Schedule the update to run every hour, for example
    schedule.every(1).hours.do(scheduled_update)  # Adjust the interval as needed

    # Start the Flask server in a separate thread
    import threading
    flask_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
    flask_thread.start()

    # Run scheduled tasks in the main thread
    while True:
        schedule.run_pending()
        time.sleep(1)
