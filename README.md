# Vector Embeddings Map

A lightweight Python library for generating, storing, and searching text embeddings using sentence transformers. This library provides an easy-to-use interface for semantic similarity search and vector storage.

## Features

- **Easy Embedding Generation**: Automatically generate embeddings from text using pre-trained models
- **Efficient Storage**: Store embeddings with associated text and metadata
- **Similarity Search**: Find semantically similar texts using cosine similarity
- **Batch Operations**: Add multiple texts efficiently in batches
- **Persistence**: Save and load embeddings to/from disk
- **Metadata Support**: Associate custom metadata with each embedding
- **Simple API**: Clean, intuitive interface for all operations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/dchampen/vector_embedds.git
cd vector_embedds
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from embeddings_map import EmbeddingsMap

# Initialize the embeddings map
em = EmbeddingsMap()

# Add texts
texts = [
    "Python is a programming language.",
    "Machine learning is fascinating.",
    "The cat sat on the mat."
]
ids = em.add_batch(texts)

# Search for similar texts
results = em.search("What is Python?", top_k=2)
for id, text, score, metadata in results:
    print(f"[Score: {score:.4f}] {text}")

# Save to disk
em.save("my_embeddings.pkl")

# Load from disk
em_loaded = EmbeddingsMap()
em_loaded.load("my_embeddings.pkl")
```

## API Reference

### Initialization

```python
EmbeddingsMap(model_name='all-MiniLM-L6-v2')
```

Create a new embeddings map instance.

**Parameters:**
- `model_name` (str): Name of the sentence-transformers model to use. Default is 'all-MiniLM-L6-v2'.

### Methods

#### `add(text, metadata=None)`

Add a single text and its embedding to the map.

**Parameters:**
- `text` (str): The text to embed and store
- `metadata` (dict, optional): Dictionary of metadata associated with this text

**Returns:** int - The unique ID assigned to this embedding

#### `add_batch(texts, metadatas=None)`

Add multiple texts efficiently in a single batch.

**Parameters:**
- `texts` (list[str]): List of texts to embed and store
- `metadatas` (list[dict], optional): List of metadata dictionaries

**Returns:** list[int] - List of unique IDs assigned to the embeddings

#### `search(query, top_k=5)`

Search for the most similar texts to the query.

**Parameters:**
- `query` (str): The query text to search for
- `top_k` (int): Number of top results to return (default: 5)

**Returns:** list[tuple] - List of tuples (id, text, similarity_score, metadata) sorted by similarity

#### `get_by_id(embedding_id)`

Retrieve a text, embedding, and metadata by ID.

**Parameters:**
- `embedding_id` (int): The ID of the embedding to retrieve

**Returns:** tuple or None - (text, embedding, metadata) or None if not found

#### `remove(embedding_id)`

Remove an embedding by ID.

**Parameters:**
- `embedding_id` (int): The ID of the embedding to remove

**Returns:** bool - True if removed successfully, False if ID not found

#### `save(filepath)`

Save the embeddings map to disk.

**Parameters:**
- `filepath` (str): Path to save the embeddings map

#### `load(filepath)`

Load an embeddings map from disk.

**Parameters:**
- `filepath` (str): Path to load the embeddings map from

#### `size()` / `__len__()`

Get the number of embeddings in the map.

**Returns:** int - Number of stored embeddings

#### `get_all()`

Get all embeddings with their IDs, texts, and metadata.

**Returns:** list[tuple] - List of tuples (id, text, metadata)

#### `clear()`

Clear all embeddings from the map.

## Example Usage

See `example.py` for a comprehensive demonstration of all features:

```bash
python example.py
```

The example demonstrates:
- Initializing an embeddings map
- Adding texts with metadata
- Performing similarity searches
- Retrieving embeddings by ID
- Saving and loading from disk
- Removing embeddings
- Listing all embeddings

## Use Cases

- **Semantic Search**: Build a search engine that understands meaning, not just keywords
- **Document Clustering**: Group similar documents together
- **Recommendation Systems**: Find similar items based on textual descriptions
- **Question Answering**: Match questions to relevant answers
- **Duplicate Detection**: Identify similar or duplicate content
- **Content Classification**: Classify texts based on semantic similarity

## Model Options

The default model is `all-MiniLM-L6-v2`, which provides a good balance of speed and accuracy. You can use any model from [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html):

```python
# Faster, smaller model
em = EmbeddingsMap(model_name='all-MiniLM-L6-v2')

# More accurate, larger model
em = EmbeddingsMap(model_name='all-mpnet-base-v2')

# Multilingual support
em = EmbeddingsMap(model_name='paraphrase-multilingual-MiniLM-L12-v2')
```

## Performance Considerations

- **Batch Operations**: Use `add_batch()` instead of multiple `add()` calls for better performance
- **Model Size**: Smaller models are faster but may be less accurate
- **Search Complexity**: Search time increases linearly with the number of embeddings
- **Memory Usage**: Each embedding is stored in memory (typically 384-768 dimensions as float32)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request