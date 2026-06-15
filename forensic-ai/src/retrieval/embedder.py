"""
Embedder

Wraps sentence-transformers to produce dense vector embeddings
for both document chunks and retrieval queries.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast inference, good semantic quality for financial text
  - ~80MB download on first use
"""

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Wraps a sentence-transformer model for encoding text to vectors.

    Usage:
        embedder = Embedder()
        vecs = embedder.embed_chunks(["text one", "text two"])
        query_vec = embedder.embed_query("revenue anomalies")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Run: pip install sentence-transformers"
            )
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded — embedding dim: {self.dim}")

    def embed_chunks(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of document chunk texts.

        Args:
            texts: List of strings to embed
            batch_size: Batch size for inference
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), dim), float32
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via dot product
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single retrieval query.

        Returns:
            numpy array of shape (dim,), float32
        """
        vec = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].astype(np.float32)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed multiple queries in batch."""
        return self.embed_chunks(queries, show_progress=False)
