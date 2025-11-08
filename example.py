"""
Example usage of the EmbeddingsMap class.

This script demonstrates the main features of the embeddings map system.
"""

from embeddings_map import EmbeddingsMap


def main():
    print("=== EmbeddingsMap Example ===\n")

    # Initialize the embeddings map
    print("1. Initializing EmbeddingsMap...")
    em = EmbeddingsMap()
    print(f"   {em}\n")

    # Add some sample texts
    print("2. Adding sample texts...")
    texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neural networks.",
        "Data science involves extracting insights from data.",
        "The cat sat on the mat.",
        "Dogs are loyal companions.",
        "JavaScript is used for web development.",
        "Cloud computing provides on-demand resources."
    ]

    # Add texts with metadata
    metadatas = [
        {"category": "programming", "language": "en"},
        {"category": "AI", "language": "en"},
        {"category": "AI", "language": "en"},
        {"category": "data", "language": "en"},
        {"category": "animals", "language": "en"},
        {"category": "animals", "language": "en"},
        {"category": "programming", "language": "en"},
        {"category": "technology", "language": "en"}
    ]

    ids = em.add_batch(texts, metadatas)
    print(f"   Added {len(ids)} texts to the map")
    print(f"   Assigned IDs: {ids}\n")

    # Perform similarity search
    print("3. Performing similarity searches...\n")

    queries = [
        "What is AI?",
        "Tell me about pets",
        "Programming languages"
    ]

    for query in queries:
        print(f"   Query: '{query}'")
        results = em.search(query, top_k=3)
        for i, (idx, text, score, metadata) in enumerate(results, 1):
            print(f"   {i}. [ID: {idx}, Score: {score:.4f}, Category: {metadata.get('category')}]")
            print(f"      {text}")
        print()

    # Get a specific embedding by ID
    print("4. Retrieving embedding by ID...")
    embedding_id = ids[0]
    result = em.get_by_id(embedding_id)
    if result:
        text, embedding, metadata = result
        print(f"   ID {embedding_id}:")
        print(f"   Text: {text}")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Metadata: {metadata}\n")

    # Save the embeddings map
    print("5. Saving embeddings map to disk...")
    em.save("embeddings_map.pkl")
    print("   Saved to 'embeddings_map.pkl'\n")

    # Load the embeddings map
    print("6. Loading embeddings map from disk...")
    em_loaded = EmbeddingsMap()
    em_loaded.load("embeddings_map.pkl")
    print(f"   Loaded: {em_loaded}")
    print(f"   Number of embeddings: {len(em_loaded)}\n")

    # Get all embeddings
    print("7. Listing all embeddings...")
    all_embeddings = em_loaded.get_all()
    print(f"   Total embeddings: {len(all_embeddings)}")
    for idx, text, metadata in all_embeddings[:3]:
        print(f"   - ID {idx}: {text[:50]}... (Category: {metadata.get('category')})")
    print()

    # Remove an embedding
    print("8. Removing an embedding...")
    remove_id = ids[4]  # "The cat sat on the mat"
    success = em_loaded.remove(remove_id)
    print(f"   Removed ID {remove_id}: {success}")
    print(f"   New size: {len(em_loaded)}\n")

    # Clear all embeddings
    print("9. Clearing all embeddings...")
    em_loaded.clear()
    print(f"   Size after clear: {len(em_loaded)}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
