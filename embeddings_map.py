"""
EmbeddingsMap - A vector embeddings storage and similarity search system.

This module provides functionality to generate, store, and search text embeddings
using sentence transformers and cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union
import pickle
import json


class EmbeddingsMap:
    """
    A class for managing text embeddings with similarity search capabilities.

    This class allows you to:
    - Generate embeddings from text using pre-trained models
    - Store embeddings with associated metadata
    - Perform similarity searches
    - Save and load embeddings to/from disk
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the EmbeddingsMap.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (fast and efficient).
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.id_to_index = {}
        self._next_id = 0

    def add(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a text and its embedding to the map.

        Args:
            text: The text to embed and store
            metadata: Optional dictionary of metadata associated with this text

        Returns:
            The unique ID assigned to this embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)

        current_id = self._next_id
        self._next_id += 1

        self.embeddings.append(embedding)
        self.texts.append(text)
        self.metadata.append(metadata or {})
        self.id_to_index[current_id] = len(self.embeddings) - 1

        return current_id

    def add_batch(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[int]:
        """
        Add multiple texts and their embeddings to the map efficiently.

        Args:
            texts: List of texts to embed and store
            metadatas: Optional list of metadata dictionaries

        Returns:
            List of unique IDs assigned to the embeddings
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("metadatas list must have same length as texts list")

        embeddings = self.model.encode(texts, convert_to_numpy=True)

        ids = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            current_id = self._next_id
            self._next_id += 1

            self.embeddings.append(embedding)
            self.texts.append(text)
            self.metadata.append(metadata)
            self.id_to_index[current_id] = len(self.embeddings) - 1
            ids.append(current_id)

        return ids

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float, Dict]]:
        """
        Search for the most similar texts to the query.

        Args:
            query: The query text to search for
            top_k: Number of top results to return

        Returns:
            List of tuples (id, text, similarity_score, metadata) sorted by similarity
        """
        if not self.embeddings:
            return []

        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.reshape(1, -1)

        embeddings_matrix = np.array(self.embeddings)
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            # Find the ID for this index
            embedding_id = None
            for eid, eidx in self.id_to_index.items():
                if eidx == idx:
                    embedding_id = eid
                    break

            results.append((
                embedding_id,
                self.texts[idx],
                float(similarities[idx]),
                self.metadata[idx]
            ))

        return results

    def get_by_id(self, embedding_id: int) -> Optional[Tuple[str, np.ndarray, Dict]]:
        """
        Retrieve a text, embedding, and metadata by ID.

        Args:
            embedding_id: The ID of the embedding to retrieve

        Returns:
            Tuple of (text, embedding, metadata) or None if not found
        """
        if embedding_id not in self.id_to_index:
            return None

        idx = self.id_to_index[embedding_id]
        return (self.texts[idx], self.embeddings[idx], self.metadata[idx])

    def remove(self, embedding_id: int) -> bool:
        """
        Remove an embedding by ID.

        Args:
            embedding_id: The ID of the embedding to remove

        Returns:
            True if removed successfully, False if ID not found
        """
        if embedding_id not in self.id_to_index:
            return False

        idx = self.id_to_index[embedding_id]

        # Remove from lists
        del self.embeddings[idx]
        del self.texts[idx]
        del self.metadata[idx]

        # Update id_to_index mapping
        del self.id_to_index[embedding_id]

        # Rebuild index mapping for all IDs after the removed one
        for eid in list(self.id_to_index.keys()):
            if self.id_to_index[eid] > idx:
                self.id_to_index[eid] -= 1

        return True

    def save(self, filepath: str):
        """
        Save the embeddings map to disk.

        Args:
            filepath: Path to save the embeddings map
        """
        data = {
            'embeddings': self.embeddings,
            'texts': self.texts,
            'metadata': self.metadata,
            'id_to_index': self.id_to_index,
            'next_id': self._next_id
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """
        Load an embeddings map from disk.

        Args:
            filepath: Path to load the embeddings map from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.id_to_index = data['id_to_index']
        self._next_id = data['next_id']

    def size(self) -> int:
        """
        Get the number of embeddings in the map.

        Returns:
            Number of stored embeddings
        """
        return len(self.embeddings)

    def clear(self):
        """Clear all embeddings from the map."""
        self.embeddings = []
        self.texts = []
        self.metadata = []
        self.id_to_index = {}
        self._next_id = 0

    def get_all(self) -> List[Tuple[int, str, Dict]]:
        """
        Get all embeddings with their IDs, texts, and metadata.

        Returns:
            List of tuples (id, text, metadata)
        """
        results = []
        for eid, idx in self.id_to_index.items():
            results.append((eid, self.texts[idx], self.metadata[idx]))
        return results

    def __len__(self) -> int:
        """Return the number of embeddings in the map."""
        return len(self.embeddings)

    def __repr__(self) -> str:
        """Return a string representation of the EmbeddingsMap."""
        return f"EmbeddingsMap(size={len(self)}, model={self.model.get_sentence_embedding_dimension()}d)"
