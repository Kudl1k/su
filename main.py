"""Hierarchical (Agglomerative) Clustering implementation using only NumPy.

Features:
  - Linkage: single, complete
  - Metrics: euclidean, manhattan
  - Returns: cluster labels for desired number of clusters, full merge history (SciPy-like linkage matrix)

Not using scikit-learn / scipy clustering routines as per assignment.

Author: (Generated with assistance)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
import numpy as np

try:  # Optional plotting dependency (user may have matplotlib per requirements.txt)
	import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
	plt = None  # type: ignore


# --------------------------------------------------------------------------------------
# Distance utilities
# --------------------------------------------------------------------------------------

def _pairwise_distance(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
	"""Compute full pairwise distance matrix between rows of X.

	Parameters
	----------
	X : np.ndarray, shape (n_samples, n_features)
	metric : {'euclidean', 'manhattan'}

	Returns
	-------
	D : np.ndarray, shape (n_samples, n_samples)
		Symmetric with zeros on diagonal.
	"""
	X = np.asarray(X, dtype=float)
	n = X.shape[0]
	if metric not in {"euclidean", "manhattan"}:
		raise ValueError(f"Unsupported metric '{metric}'. Use 'euclidean' or 'manhattan'.")

	# Efficient vectorized computation
	if metric == "euclidean":
		# (x - y)^2 = x^2 + y^2 - 2xy
		sq_norms = np.sum(X ** 2, axis=1)
		D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)
		# Numerical errors may cause tiny negatives
		np.maximum(D2, 0.0, out=D2)
		D = np.sqrt(D2, out=D2)
	else:  # Manhattan
		# Broadcasting difference: O(n^2 * d) memory if done naively; do in blocks if large.
		# For teaching / small datasets, simple implementation is fine.
		D = np.zeros((n, n), dtype=float)
		for i in range(n):
			diff = np.abs(X[i] - X)  # (n, d)
			D[i] = np.sum(diff, axis=1)
		# Symmetric by construction; diagonal already zero.

	# Ensure exact zeros on diagonal
	np.fill_diagonal(D, 0.0)
	return D


# --------------------------------------------------------------------------------------
# Core HAC implementation
# --------------------------------------------------------------------------------------

@dataclass
class MergeRecord:
	c1: int  # id of first merged cluster
	c2: int  # id of second merged cluster
	dist: float  # distance at merge
	size: int  # resulting cluster size


def agglomerative_clustering(
	X: np.ndarray,
	n_clusters: int = 2,
	linkage: str = "single",
	metric: str = "euclidean",
	return_linkage_matrix: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	"""Perform Hierarchical Agglomerative Clustering.

	Computes the full merge history (n_samples-1 merges) and then derives labels by
	cutting after (n_samples - n_clusters) merges.

	Parameters
	----------
	X : np.ndarray, shape (n_samples, n_features)
	n_clusters : int
		Desired number of clusters (1 .. n_samples).
	linkage : {'single', 'complete'}
		Linkage strategy.
	metric : {'euclidean', 'manhattan'}
		Distance metric for point-wise distances.
	return_linkage_matrix : bool
		If True, also return SciPy-like linkage matrix of shape (n_samples-1, 4).

	Returns
	-------
	labels : np.ndarray, shape (n_samples,)
		Integer labels 0..n_clusters-1.
	Z : np.ndarray | None
		Linkage matrix: each row [c1, c2, dist, size]. Cluster ids < n_samples refer to original points;
		>= n_samples are newly formed clusters in order of appearance. None if return_linkage_matrix=False.
	"""
	X = np.asarray(X, dtype=float)
	n_samples = X.shape[0]
	if n_samples == 0:
		raise ValueError("Empty dataset.")
	if not 1 <= n_clusters <= n_samples:
		raise ValueError("n_clusters must be within [1, n_samples].")
	if linkage not in {"single", "complete"}:
		raise ValueError("Unsupported linkage. Use 'single' or 'complete'.")

	# Initial distance matrix between individual points
	base_dist = _pairwise_distance(X, metric=metric)

	# Active clusters: list of sets of sample indices
	clusters: List[set[int]] = [ {i} for i in range(n_samples) ]
	cluster_ids: List[int] = list(range(n_samples))  # External IDs matching SciPy style
	sizes = [1] * n_samples

	# Distance matrix between active clusters (initialized as point distances)
	# We'll copy to avoid modifying base_dist in place unexpectedly.
	D = base_dist.copy()
	np.fill_diagonal(D, np.inf)  # inf to ignore self-merges

	merges: List[MergeRecord] = []
	next_cluster_id = n_samples

	# Helper to update distance row for new cluster (last index) using rows i and j (before deletion)
	def _update_distances(D: np.ndarray, idx_a: int, idx_b: int, linkage: str) -> None:
		# New cluster is at last index (after building new D outside this function)
		new_idx = D.shape[0] - 1
		if linkage == "single":
			new_row = np.minimum(D[idx_a, :-1], D[idx_b, :-1])
		else:  # complete
			new_row = np.maximum(D[idx_a, :-1], D[idx_b, :-1])
		D[new_idx, :-1] = new_row
		D[:-1, new_idx] = new_row
		D[new_idx, new_idx] = np.inf

	# We will rebuild D each iteration with new cluster at end; simpler & clear for teaching.
	while len(clusters) > 1:
		# Find closest pair (argmin of D)
		a, b = divmod(np.argmin(D), D.shape[1])
		if a == b:
			# Should not happen because diagonal is inf, but safety check
			raise RuntimeError("Failed to find valid pair to merge.")
		if b < a:  # ensure a < b for consistency
			a, b = b, a

		c1_id, c2_id = cluster_ids[a], cluster_ids[b]
		dist_ab = D[a, b]
		new_size = sizes[a] + sizes[b]

		# Record merge
		merges.append(MergeRecord(c1=c1_id, c2=c2_id, dist=float(dist_ab), size=new_size))

		# Create new cluster
		new_cluster = clusters[a] | clusters[b]

		# Build new distance matrix with rows/cols except a,b plus new cluster
		mask = np.ones(len(clusters), dtype=bool)
		mask[[a, b]] = False
		# Extract the submatrix of remaining clusters
		D_sub = D[mask][:, mask]

		# Append placeholder row/col for new cluster (inf distances initially)
		m = D_sub.shape[0]
		D_new = np.full((m + 1, m + 1), np.inf, dtype=float)
		if m > 0:
			D_new[:m, :m] = D_sub

		# Update cluster tracking structures
		remaining_clusters = [clusters[i] for i in range(len(clusters)) if mask[i]]
		remaining_ids = [cluster_ids[i] for i in range(len(clusters)) if mask[i]]
		remaining_sizes = [sizes[i] for i in range(len(clusters)) if mask[i]]

		# Prepare temporary D with old indices for a,b relative to mask for updating distances
		# To reuse our simple update logic, we create a temporary matrix referencing old rows.
		# Simpler: compute distances via linkage rule from original D directly.
		if len(remaining_clusters) > 0:
			# We need distances from new cluster to each remaining cluster
			if linkage == "single":
				new_dists = np.minimum(D[a, mask], D[b, mask])
			else:
				new_dists = np.maximum(D[a, mask], D[b, mask])
			D_new[m, :m] = new_dists
			D_new[:m, m] = new_dists
		D_new[m, m] = np.inf

		# Commit structures
		clusters = remaining_clusters + [new_cluster]
		cluster_ids = remaining_ids + [next_cluster_id]
		sizes = remaining_sizes + [new_size]
		next_cluster_id += 1
		D = D_new

	# Build linkage matrix Z
	if merges:
		Z = np.array([[mr.c1, mr.c2, mr.dist, mr.size] for mr in merges], dtype=float)
	else:  # Single point dataset
		Z = np.zeros((0, 4), dtype=float)

	# Derive labels for requested n_clusters by applying first (n_samples - n_clusters) merges
	if n_clusters == n_samples:
		labels = np.arange(n_samples)
	else:
		parent = {i: i for i in range(n_samples)}  # union-find parent for original samples only
		sizeUF = {i: 1 for i in range(n_samples)}

		def find(x):
			while parent[x] != x:
				parent[x] = parent[parent[x]]
				x = parent[x]
			return x

		next_virtual = n_samples  # virtual ids for merged clusters (we only need mapping for find)
		# We'll map virtual cluster id to one of its children to allow root resolution of original samples.
		# Simpler: maintain mapping cluster_id -> representative original sample inside it.
		rep = {i: i for i in range(n_samples)}  # representative original sample for each cluster id (orig or virtual)

		merges_to_apply = merges[: n_samples - n_clusters]
		for mr in merges_to_apply:
			r1 = find(rep[mr.c1]) if mr.c1 >= n_samples else find(mr.c1)
			r2 = find(rep[mr.c2]) if mr.c2 >= n_samples else find(mr.c2)
			if r1 == r2:
				continue
			# Union by size
			if sizeUF[r1] < sizeUF[r2]:
				r1, r2 = r2, r1
			parent[r2] = r1
			sizeUF[r1] += sizeUF[r2]
			# Create virtual id mapping to representative
			rep[next_virtual] = r1
			next_virtual += 1

		# Final label assignment
		roots = [find(i) for i in range(n_samples)]
		unique_roots, inverse = np.unique(roots, return_inverse=True)
		labels = inverse  # already 0..k-1

	return labels, (Z if return_linkage_matrix else None)


# --------------------------------------------------------------------------------------
# Visualization helper
# --------------------------------------------------------------------------------------

def plot_clusters(X: np.ndarray, labels: np.ndarray, title: str = "Clusters") -> None:
	"""Scatter plot for 2D data with cluster coloring.

	Does nothing if matplotlib isn't available or dimensionality != 2.
	"""
	if plt is None:
		print("matplotlib not available; skipping plot.")
		return
	if X.shape[1] != 2:
		print("plot_clusters: Only supports 2D data (found shape %s)." % (X.shape,))
		return
	plt.figure(figsize=(6, 5))
	scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=50, edgecolor="k")
	plt.title(title)
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.grid(True, alpha=0.3)
	plt.colorbar(scatter, label="Cluster")
	plt.tight_layout()
	plt.show()


# --------------------------------------------------------------------------------------
# Example usage when run as a script
# --------------------------------------------------------------------------------------

def _demo():
	rng = np.random.default_rng(42)
	# Create simple 2D dataset: five blobs
	A = rng.normal(loc=(-4, -2), scale=0.6, size=(25, 2))
	B = rng.normal(loc=(0, 3), scale=0.5, size=(25, 2))
	C = rng.normal(loc=(4, -1), scale=0.7, size=(25, 2))
	D = rng.normal(loc=(6, 4), scale=0.5, size=(25, 2))
	E = rng.normal(loc=(-6, 5), scale=0.7, size=(25, 2))

	X = np.vstack([A, B, C, D, E])

	labels_single_euclid, Z1 = agglomerative_clustering(
		X, n_clusters=5, linkage="single", metric="euclidean"
	)
	labels_complete_manhattan, Z2 = agglomerative_clustering(
		X, n_clusters=5, linkage="complete", metric="manhattan"
	)

	print("Single linkage (Euclidean) labels:", labels_single_euclid)
	print("Complete linkage (Manhattan) labels:", labels_complete_manhattan)
	print("Linkage matrix shape (single/euclid):", Z1.shape)

	# Visualize
	plot_clusters(X, labels_single_euclid, title="Single Linkage (Euclidean)")
	plot_clusters(X, labels_complete_manhattan, title="Complete Linkage (Manhattan)")


if __name__ == "__main__":  # pragma: no cover
	_demo()

