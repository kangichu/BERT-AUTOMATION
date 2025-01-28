## Comprehensive Documentation for Real Estate Data Pipeline

### Table of Contents
1. **Overview**
2. **Key Concepts**
3. **Pipeline Steps**
   - Data Retrieval
   - Narrative Generation
   - Embedding Generation
   - Storage for Search
   - Visualization
4. **Code Walkthrough**
   - File Structure
   - Main Pipeline (`main.py`)
   - Supporting Modules
5. **Logging and Error Handling**
6. **FAQs and Troubleshooting

---

### 1. Overview
This pipeline is designed to process real estate data, transforming raw listings from a MySQL database into insights using advanced NLP techniques and embedding visualizations. The steps include:

1. **Data Retrieval**: Connect to a database and extract relevant data.
2. **Narrative Generation**: Use AI models like GPT-2 and BLOOM to create engaging property descriptions.
3. **Embedding Creation**: Convert narratives into vector representations with BERT.
4. **Storage**: Index embeddings for fast similarity searches using FAISS.
5. **Visualization**: Present data interactively in TensorBoard.

---

### 2. Key Concepts

#### **Transformers**
- **Definition**: Neural networks that use attention mechanisms for context-aware processing.
- **Usage**:
  - GPT-2: For generating property narratives.
  - BERT: For embedding text into dense vectors.

#### **Tokenizers**
- **Definition**: Tools that break text into smaller units, like words or subwords (tokens), which can be understood by transformer models.
- **Types**:
  - Word-based: Splits text by spaces.
  - Subword-based: Splits words into smaller meaningful units.
- **Example**:
  - Input: "Real estate listings"
  - Tokenized Output: ["Real", "estate", "listings"]
- **Importance**: Tokenizers are essential for preparing raw text so models like GPT-2 and BERT can process it efficiently.

#### **Embeddings**
- **Definition**: Dense vector representations of text that encode semantic meaning.
- **Applications**:
  - Similarity search (e.g., finding properties similar to a query).

#### **FAISS**
- **Definition**: A high-performance library for similarity search.
- **Purpose**: Efficiently stores and queries embeddings.

#### **TensorBoard**
- **Definition**: A visualization tool for ML metrics and embeddings.
- **Purpose**: Inspect patterns and relationships in the embeddings.

---


### 3. Pipeline Steps

#### **Step 1: Data Retrieval**
- **What Happens**:
  - Connects to a MySQL database.
  - Extracts published listings with a JOIN query to enrich data with complex details.

- **Code Snippet**:
```python
from pymysql import connect

def fetch_data_from_mysql():
    query = """
    SELECT listings.*, complexes.title AS complex_title
    FROM listings
    LEFT JOIN complexes ON listings.complex_id = complexes.id
    WHERE listings.status = 'Published';
    """
    rows = cursor.fetchall()
    return rows
```

---

#### **Step 2: Narrative Generation**
- **What Happens**:
  - Generates narratives using GPT-2, BLOOM, and Llama.
  - Constructs prompts to guide AI models in producing relevant content.

- **Key Functions**:
  - `generate_narratives_with_gpt`
  - `generate_narratives_with_bloom`
  - `generate_narratives_with_llama`

- **Code Snippet**:
```python
prompt = f"Create a listing description for {name} in {location} priced at {price}."
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

#### **Step 3: Embedding Generation**
- **What Happens**:
  - Converts narratives into vector embeddings using BERT.
  - Applies tokenization and processes text through BERT’s layers.

- **Code Snippet**:
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokens = tokenizer(narrative, return_tensors="pt", truncation=True)
output = model(**tokens)
embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
```

---

#### **Step 4: Storage for Search**
- **What Happens**:
  - Embeddings are added to a FAISS index for similarity search.

- **Code Snippet**:
```python
import faiss

manager = FAISSManager(dimension=768)
manager.add_vectors(vector_data)
manager.save_index()
```

---

#### **Step 5: Visualization**
- **What Happens**:
  - Embeddings and metadata are exported to TensorBoard.
  - TensorBoard's projector visualizes embeddings in a reduced-dimensional space.

- **Code Snippet**:
```python
create_projector_config(
    log_dir="logs/embedding_logs",
    embeddings=embeddings,
    metadata=metadata
)
```

---

### 4. Code Walkthrough

#### **File Structure**
```plaintext
project/
│
├── main.py                   # Main pipeline orchestration
├── data_fetcher/
│   └── mysql_fetcher.py      # MySQL interactions
├── narrative_generator/
│   └── gpt_narrative.py      # GPT-2, BLOOM, Llama narrative generation
├── vectorizer/
│   └── bert_vectorizer.py    # BERT embeddings
├── database/
│   └── faiss_store.py        # FAISS storage
├── data_handler/
│   ├── tensorboard_exporter.py  # TensorBoard export
├── utils/
│   └── logger.py             # Logging setup
├── config.py                 # Database configuration
└── logs/                     # Log files
```

---

### 5. Logging and Error Handling

#### **Logging**
- Logs are stored in the `logs/` directory.
- Errors are logged with timestamps for debugging.

#### **Error Handling**
- Each step uses `try-except` blocks.
- Example:
```python
try:
    data = fetch_data_from_mysql()
except Exception as e:
    logging.error(f"Failed to fetch data: {e}")
```

---

### 6. FAQs and Troubleshooting

**Q: Why is TensorBoard not showing data?**
- Ensure the `projector_config.pbtxt` and `.tsv` files are correctly exported.

**Q: How do I add new models?**
- Extend `narrative_generator.py` with new functions for loading and using other models.

**Q: Can I use cloud-hosted FAISS?**
- Yes, by integrating FAISS with a database like Redis or Milvus.

---

