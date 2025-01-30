import logging
import os
import sys
import schedule
import time
import threading
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

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
        store_embeddings_in_faiss(vector_data, 'logs/embeddings/metadata.tsv')

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

def search_faiss(query):
    try:
        logging.info(f"Starting FAISS search for query: {query}")

        # Initialize the FAISS manager with the correct dimension and narratives
        logging.info("Loading FAISS index...")
        narratives = [...]  # You need to provide your list of narratives here
        faiss_manager = FAISSManager(dimension=768, index_dir="faiss_index", narratives=narratives)

        faiss_manager.load_index()  # Load the index
        logging.info("FAISS index loaded successfully.")

        # Generate embedding for the query
        logging.info("Generating query embedding...")
        query_embedding = generate_single_embedding(query)
        logging.info(f"Query embedding generated: {query_embedding[:5]}... (truncated)")

       # Perform the search to get indices and distances
        logging.info("Performing FAISS search...")
        D, I = faiss_manager.search(query_embedding, k=5)

        logging.info(f"Search completed. Indices: {I}, Distances: {D}")

        # Check if search returned valid indices
        if len(I) == 0 or all(idx == -1 for idx in I):
            logging.warning("Search returned no valid indices.")
            return []

        # Check the types and the shape of I and D
        logging.info(f"Types: Indices: {type(I)}, Distances: {type(D)}")
        logging.info(f"Shapes: Indices: {I.shape}, Distances: {D.shape}")

        # Ensure all indices are integers, whether I is a scalar or array
        if isinstance(I, np.ndarray):
            indices_to_reconstruct = I.astype(int).tolist()
        else:
            indices_to_reconstruct = [int(I)]

        # Retrieve narratives from search results
        logging.info("Retrieving narratives from search results...")

        # Ensure there are valid indices and handle potential empty or invalid values
        valid_indices = [i for i in indices_to_reconstruct if i >= 0 and i < len(narratives)]

        if not valid_indices:
            logging.warning("No valid indices found for reconstruction.")
            return []

        results = []
        for idx in valid_indices:
            try:
                if isinstance(idx, int) and idx >= 0:
                    logging.debug(f"Attempting to reconstruct index {idx}")
                    result = faiss_manager.reconstruct(idx)
                    logging.debug(f"Reconstructed result for index {idx}: {result}")
                    if result is None:
                        logging.warning(f"Reconstruction for index {idx} returned None.")
                    else:
                        results.append(result)
                else:
                    logging.warning(f"Skipping invalid index: {idx}")
            except Exception as e:
                logging.error(f"Failed to reconstruct index {idx}: {e}")

        # Log the results before extracting narratives
        logging.info(f"Reconstruction results: {results}")

        # Extract narratives from the results
        retrieved_narratives = [result[1] for result in results if len(result) > 1]

        # If there are no valid narratives
        if not retrieved_narratives:
            logging.warning("No valid narratives retrieved from the search results.")

        logging.info(f"Retrieved {len(retrieved_narratives)} relevant narratives.")
        logging.info(f"Retrieved narratives: {retrieved_narratives}")
        return retrieved_narratives

    except Exception as e:
        logging.critical(f"Search FAISS failed: {e}", exc_info=True)
        raise

def generate_response(query, retrieved_narratives):
    try:
        logging.info('Generating Response...')
        # Prepare the prompt to send to GPT
        prompt = f"Answer the following question based on the narratives:\n\nQuestion: {query}\n\n"
        prompt += "Here are some relevant narratives:\n" + "\n".join(retrieved_narratives)
        
        logging.info(f"Prompt for GPT: {prompt[:200]}... (truncated)")  # Log first 200 characters

        # Generate the response using GPT
        response = generate_response_with_gpt(prompt)  # This is where GPT is used
        logging.info(f"Generated response: {response[:100]}... (truncated)")  # Log first 100 chars
        return response
    except Exception as e:
        logging.error(f"Generate response failed: {e}")
        raise

def scheduled_update():
    update_pipeline_with_new_listings('faiss_index')
    logging.info("Scheduled update completed.")

@app.route('/search', methods=['POST'])
def search():
    if request.is_json:
        try:
            data = request.get_json()
            logging.info(f"Parsed JSON data: {data}")
            if not data or 'query' not in data:
                return jsonify({"error": "No query provided"}), 400
            
            user_query = data['query']
            index_path = 'faiss_index'  # Ensure the file has a proper .index extension
            logging.info(f"FAISS index path: {os.path.abspath(index_path)}")
            
            logging.info(f"Received query: {user_query}")
            
            if not os.path.exists(index_path):
                logging.error(f"FAISS index file {index_path} does not exist.")
                return jsonify({"error": "FAISS index not found"}), 500
            
            retrieved_narratives = search_faiss(user_query)
            
            logging.info(f"Retrieved narratives: {retrieved_narratives}")
            
            # Generate response
            if retrieved_narratives:
                response = generate_response(user_query, retrieved_narratives)
                logging.info(f"Generated Response: {response}")
                return jsonify({"response": response}), 200
            else:
                logging.info("No relevant results found.")
                return jsonify({"response": "No relevant results found."}), 200
        except Exception as e:
            logging.error(f"Error during search or response generation: {str(e)}")
            return jsonify({"error": f"An error occurred while processing your request: {str(e)}"}), 500
    else:
        logging.info("Request was not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
    

@app.route('/update_pipeline', methods=['POST'])
def update_pipeline_endpoint():
    try:
        update_pipeline_with_new_listings('faiss_index')
        return jsonify({"status": "Update completed successfully"}), 200
    except Exception as e:
        logging.error(f"Error during update: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "pong"}), 200

if __name__ == "__main__":
    setup_logging()
    print("\n🚀 Real Estate Data Pipeline is now running...\n")
    
    if '--run-pipeline' in sys.argv:
        print("🔄 Running initial pipeline execution...\n")
        run_pipeline()

        # Optionally, you might want to exit after running the pipeline
        print("Pipeline execution completed. Exiting.")
        sys.exit(0)
    
    # Schedule the update to run every hour, for example
    schedule.every(1).hours.do(scheduled_update)  # Adjust the interval as needed

    # Add these configuration settings before running the Flask app
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.config['JSON_AS_ASCII'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB, adjust if needed

    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 7000, 'use_reloader': False, 'debug': True}, daemon=True)
    flask_thread.start()

    print("🌍 Flask API is live at http://localhost:7000\n")
    print("🕒 Scheduler is running. Checking for updates every hour...\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Stopping the server and scheduler...")
        flask_thread.join(timeout=1)  # Wait for Flask thread to stop gracefully
        sys.exit(0)