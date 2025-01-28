
## Comprehensive Documentation for Real Estate Data Pipeline

### Table of Contents
1. Overview  
2. Key Concepts  
3. Execution Flow (Detailed Pipeline Steps)  
   - Data Retrieval  
   - Narrative Generation  
   - Embedding Generation  
   - FAISS Indexing and Storage  
   - Visualization with TensorBoard  
   - Updating the FAISS Index with New Listings  
4. Code Walkthrough  
   - File Structure  
   - Function Descriptions  
5. Logging and Error Handling  
6. FAQs and Troubleshooting  

---

### 1. Overview  
This real estate data pipeline processes property listings from a MySQL database, generating narratives, creating vector embeddings, and indexing them for efficient similarity-based searches. Key functionalities include:

- Fetching real-time property data from MySQL.
- Generating narrative descriptions using AI or manual methods.
- Converting descriptions into vector embeddings using BERT.
- Storing and searching embeddings via FAISS for similarity.
- Visualizing embeddings with TensorBoard for insights.
- Updating the index with new listings periodically.

The system also provides Flask API endpoints for real-time searches and data updates.

---

### 2. Key Concepts  

**Transformers**: 
- **Models**: 
  - **GPT-2, BLOOM**: Used for generating property descriptions.
  - **BERT**: Converts text to vector embeddings for similarity analysis.

**FAISS**: 
- **Purpose**: Facilitates high-speed similarity searches by indexing embeddings.

**TensorBoard**: 
- **Function**: Visualizes embeddings and their metadata for analytical purposes.

**Flask API**: 
- **Endpoints**: 
  - `/search`: Allows searching for properties similar to given text queries.
  - `/update_pipeline`: Endpoint to integrate new listings into the FAISS index.

---

### 3. Execution Flow  

#### **Pipeline Execution Flow in `main.py`**

Here's how the pipeline works step-by-step:

---

#### **Step 1: Data Retrieval**

**Purpose**: 
- Collects all published property listings from MySQL, including complex details.

**Key Function**: 
```python
def fetch_data_from_mysql():
    """
    Query MySQL to get all published listings with complex details.
    """
    query = """
    SELECT listings.*, complexes.title AS complex_title
    FROM listings
    LEFT JOIN complexes ON listings.complex_id = complexes.id
    WHERE listings.status = 'Published';
    """
    cursor.execute(query)
    return cursor.fetchall()

Pipeline Code: 
python
rows = fetch_data_from_mysql()
logging.info(f"Retrieved {len(rows)} listings from MySQL.")

Step 2: Narrative Generation
Purpose: 
Creates narrative descriptions for each property listing.

Key Function (Manual):  
python
def generate_narrative_manually(rows):
    """
    Generates simple property descriptions based on data fields.
    """
    narratives = []
    for row in rows:
        narrative = f"{row['name']} is a {row['listing_type']} in {row['county_specific']}."
        narratives.append((row['id'], narrative))
    return narratives

Pipeline Code: 
python
narratives = generate_narrative_manually(rows)
logging.info(f"Generated {len(narratives)} property narratives.")

Step 3: Embedding Generation
Purpose: 
Converts narrative texts into vector embeddings for similarity searches.

Key Function:  
python
def generate_embeddings(narratives):
    """
    Uses BERT to generate embeddings from narrative texts.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    vector_data = []
    for id, narrative in narratives:
        tokens = tokenizer(narrative, return_tensors="pt")
        output = model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
        vector_data.append((id, narrative, embedding))
    return vector_data

Pipeline Code: 
python
vector_data = generate_embeddings(narratives)
logging.info(f"Generated embeddings for {len(vector_data)} narratives.")

Step 4: FAISS Indexing and Storage
Purpose: 
Stores the generated embeddings in FAISS for efficient searching.

Key Function: 
python
def store_embeddings_in_faiss(vector_data):
    """
    Adds embeddings to a FAISS index, initializing if not exists.
    """
    faiss_manager = FAISSManager()
    faiss_manager.add_vectors(vector_data)
    logging.info(f"Stored {len(vector_data)} embeddings in the FAISS index.")

Pipeline Code: 
python
store_embeddings_in_faiss(vector_data)
logging.info("Embeddings stored in FAISS successfully.")

Step 5: Visualization with TensorBoard
Purpose: 
Exports embeddings for visualization in TensorBoard.

Key Function: 
python
def export_to_tensorboard(embeddings, metadata, log_dir="logs/embedding_logs"):
    """
    Prepares embeddings and metadata for TensorBoard visualization.
    """
    os.makedirs(log_dir, exist_ok=True)
    np.savetxt(os.path.join(log_dir, "embeddings.tsv"), embeddings, delimiter="\t")
    with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
        f.write("\n".join(metadata))
    logging.info("Embeddings exported for TensorBoard.")

Pipeline Code: 
python
export_to_tensorboard(
    [item[2] for item in vector_data],
    [f"ID: {item[0]} | Narrative: {item[1]}" for item in vector_data]
)

Step 6: Updating the FAISS Index with New Listings
Purpose: 
Processes only new listings to update the FAISS index periodically.

Key Function: 
python
def update_pipeline_with_new_listings(index_path):
    """
    Retrieves new listings, generates their embeddings, and updates the FAISS index.
    """
    last_processed_id = load_last_processed_id()
    new_rows, new_last_processed_id = fetch_new_listings_from_mysql(last_processed_id)
    if new_rows:
        narratives = generate_narrative_manually(new_rows)
        vector_data = generate_embeddings(narratives)
        faiss_manager = FAISSManager()
        faiss_index = faiss_manager.load_index(index_path)
        for item in vector_data:
            faiss_index.add(np.array([item[2]]))
        faiss_manager.save_index(faiss_index, index_path)
        save_last_processed_id(new_last_processed_id)
        logging.info(f"Updated FAISS index with {len(new_rows)} new listings.")

Pipeline Code: 
python
update_pipeline_with_new_listings('faiss_index')

4. Code Walkthrough
File Structure
project/
├── main.py                   # Main pipeline orchestration
├── data_fetcher/
│   ├── mysql_fetcher.py      # Fetch data from MySQL
│   └── last_processed.py     # Track last processed listing ID
├── narrative_generator/
│   ├── gpt_narrative.py      # AI-powered narrative generation
│   └── manual_narrative.py   # Manual narrative logic
├── vectorizer/
│   └── bert_vectorizer.py    # Embedding generation
├── database/
│   └── faiss_store.py        # FAISS index management
├── data_handler/
│   └── tensorboard_exporter.py  # TensorBoard export
├── utils/
│   └── logger.py             # Logging setup
└── logs/                     # Log files

5. Logging and Error Handling
Logging Setup: 
python
def setup_logging():
    """
    Configures daily rotating log files.
    """
    handler = logging.handlers.TimedRotatingFileHandler("logs/pipeline.log", when="midnight")
    logging.basicConfig(handlers=[handler], level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Logging initialized.")

Error Handling Example: 
python
try:
    rows = fetch_data_from_mysql()
except Exception as e:
    logging.error(f"Error during data retrieval: {e}")

6. FAQs and Troubleshooting
Q: How do I handle new listings?  
Use update_pipeline_with_new_listings('faiss_index').

Q: Why are embeddings not visible in TensorBoard?  
Verify that embeddings are being exported to logs/embedding_logs.

Q: How do I add new models?  
Modify or add to the narrative_generator module to incorporate new AI models for narrative generation.

This documentation should now provide a clearer, more detailed understanding of the code's functionality and structure within the real estate data pipeline.