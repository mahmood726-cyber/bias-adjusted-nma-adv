"""Graph Convolutional Network (GCN) treatment embedding and regularization module."""

from __future__ import annotations

import numpy as np

class TreatmentGCNRegularizer:
    """NumPy-based GCN layer to compute treatment embeddings and topological regularizers."""

    def __init__(self, embedding_dim: int = 4, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        self.weights: np.ndarray | None = None

    def fit_transform(
        self,
        treatments: list[str],
        edges: list[tuple[str, str]],
        features: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute treatment embeddings using a 1-layer GCN.

        Returns an embedding matrix of shape (n_treatments, embedding_dim).
        """
        n_nodes = len(treatments)
        trt_to_idx = {t: idx for idx, t in enumerate(treatments)}

        # 1. Build Adjacency Matrix with self-loops
        a = np.eye(n_nodes, dtype=float)
        for u, v in edges:
            if u in trt_to_idx and v in trt_to_idx:
                idx_u, idx_v = trt_to_idx[u], trt_to_idx[v]
                a[idx_u, idx_v] = 1.0
                a[idx_v, idx_u] = 1.0

        # 2. Symmetric Normalization D^-1/2 * A * D^-1/2
        degrees = np.sum(a, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-6)))
        a_norm = d_inv_sqrt @ a @ d_inv_sqrt

        # 3. Node Features (identity matrix if not provided)
        if features is None:
            features = np.eye(n_nodes, dtype=float)

        # 4. GCN Weights Initialization
        if self.weights is None:
            self.weights = self.rng.normal(0.0, 0.1, size=(features.shape[1], self.embedding_dim))

        # 5. Forward propagation: Activation(A_norm * X * W)
        support = features @ self.weights
        embeddings = a_norm @ support
        # ReLU activation
        return np.maximum(embeddings, 0.0)

    def get_topological_precision_matrix(
        self,
        treatments: list[str],
        edges: list[tuple[str, str]],
        lambda_reg: float = 0.1
    ) -> np.ndarray:
        """Compute the GCN-derived topological prior precision matrix.

        Shrinks treatments that are far apart in the GCN embedding space.
        """
        n_nodes = len(treatments)
        embeddings = self.fit_transform(treatments, edges)

        # Pairwise Euclidean distance matrix in GCN space
        p_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(n_nodes):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                p_matrix[i, j] = lambda_reg * np.exp(-dist)

        # Normalize rows to create a proper precision matrix (Laplacian-style)
        for i in range(n_nodes):
            off_diag_sum = np.sum(p_matrix[i, :]) - p_matrix[i, i]
            p_matrix[i, :] = -p_matrix[i, :]
            p_matrix[i, i] = off_diag_sum

        return p_matrix
