"""
Preprocessing wrapper for Graph-Conditioned InterventionalPFN models.

This module extends GraphConditionedInterventionalPFNSklearn with automatic
preprocessing of input data to match the training configuration.

Key features:
- Automatic feature truncation/padding to match model's expected feature count
- Outlier removal (clipping to quantiles)
- Feature standardization
- Target scaling to [-1, 1] range
- Sample truncation/padding to match model's expected sample counts
- Automatic adjacency matrix construction for causal inference

Usage:
    wrapper = PreprocessingGraphConditionedPFN(
        config_path="config.yaml",
        checkpoint_path="model.pt"
    )
    wrapper.load()
    
    # Fit preprocessing on training data
    wrapper.fit(X_train, T_train, Y_train)
    
    # Prediction with automatic preprocessing
    preds = wrapper.predict(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv,
        adjacency_matrix=None,  # Auto-constructed if None
        prediction_type="mean"
    )
"""

from __future__ import annotations
from typing import Optional, Any, Literal, Dict, Tuple
import numpy as np
import yaml
from sklearn.cluster import KMeans

# Import base class
try:
    from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
except ImportError:
    try:
        from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
    except ImportError:
        from GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn


class PreprocessingGraphConditionedPFN(GraphConditionedInterventionalPFNSklearn):
    """
    Graph-Conditioned InterventionalPFN with automatic preprocessing.
    
    This class extends GraphConditionedInterventionalPFNSklearn to automatically
    apply the same preprocessing that was used during training. This includes:
    
    - Feature truncation/padding to match model's num_features
    - Outlier removal via quantile clipping
    - Feature standardization (zero mean, unit variance)
    - Target scaling to [-1, 1] range
    - Sample truncation/padding to match model's expected sample counts
    
    The preprocessing parameters are extracted from the model's config file.
    
    Parameters:
        config_path (str): Path to YAML config file
        checkpoint_path (str): Path to model checkpoint
        device (str, optional): Device for inference ('cpu', 'cuda', etc.)
        verbose (bool): Print detailed information
        use_clustering (bool): If True, use adaptive clustering when train samples exceed max_n_train
        random_state (int, optional): Random seed for clustering reproducibility
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        use_clustering: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            verbose=verbose,
        )
        
        # Preprocessing config (loaded from config file)
        self.preprocessing_config: Dict[str, Any] = {}
        self.dataset_config: Dict[str, Any] = {}
        
        # Clustering parameters
        self.use_clustering = use_clustering
        self.random_state = random_state
        
        # Test feature masking (loaded from config, default 0.0 = no masking)
        self.test_feature_mask_fraction: float = 0.0
        
        # Fitted preprocessing parameters (computed from training data)
        self._fitted = False
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._lower_bounds: Optional[np.ndarray] = None
        self._upper_bounds: Optional[np.ndarray] = None
        self._y_min: float = 0.0
        self._y_max: float = 1.0
        self._y_range: float = 1.0
        self._n_original_features: int = 0
        self._n_real_features: int = 0  # Number of real (non-padded) features
        
    def _get_config_value(self, config_dict: Dict, key: str, default: Any = None) -> Any:
        """Extract value from config entry that may be plain or dict with 'value' key."""
        raw = config_dict.get(key, default)
        if isinstance(raw, dict) and "value" in raw:
            return raw["value"]
        return raw
    
    def load(self, override_kwargs: Optional[dict[str, Any]] = None) -> "PreprocessingGraphConditionedPFN":
        """
        Load config, build model, load checkpoint, and extract preprocessing config.
        
        Args:
            override_kwargs: Optional dict to override config parameters
            
        Returns:
            self for method chaining
        """
        # Call parent load
        super().load(override_kwargs)
        
        # Load preprocessing and dataset config from config file
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.preprocessing_config = config.get('preprocessing_config', {})
            self.dataset_config = config.get('dataset_config', {})
            
            # Extract test feature mask fraction from dataset config
            self.test_feature_mask_fraction = self._get_config_value(
                self.dataset_config, 'test_feature_mask_fraction', 0.0
            )
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Loaded preprocessing config:")
                print(f"  remove_outliers: {self._get_config_value(self.preprocessing_config, 'remove_outliers', True)}")
                print(f"  outlier_quantile: {self._get_config_value(self.preprocessing_config, 'outlier_quantile', 0.99)}")
                print(f"  feature_standardize: {self._get_config_value(self.preprocessing_config, 'feature_standardize', True)}")
                print(f"  test_feature_mask_fraction: {self.test_feature_mask_fraction}")
        
        return self
    
    def fit(
        self,
        X: np.ndarray,
        T: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> "PreprocessingGraphConditionedPFN":
        """
        Fit preprocessing parameters on training data.
        
        This computes the statistics needed for preprocessing:
        - Outlier bounds (quantiles) from X
        - Feature mean and std from X (after outlier removal)
        - Target range from Y (for scaling to [-1, 1])
        
        Args:
            X: Training features, shape (N, L)
            T: Training treatment, shape (N,) or (N, 1) - optional, not used for fitting
            Y: Training targets, shape (N,) or (N, 1) - used for target scaling
            
        Returns:
            self for method chaining
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        X = np.asarray(X, dtype=np.float32)
        self._n_original_features = X.shape[1]
        
        # Get model's expected number of features
        model_n_features = self.model.num_features
        
        # Truncate features if needed (but DON'T pad yet - compute stats on real features only)
        if X.shape[1] > model_n_features:
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Truncating features from {X.shape[1]} to {model_n_features}")
            X = X[:, :model_n_features]
            n_real_features = model_n_features
        else:
            n_real_features = X.shape[1]
            if self.verbose and X.shape[1] < model_n_features:
                print(f"[PreprocessingGraphConditionedPFN] Will pad features from {X.shape[1]} to {model_n_features} after preprocessing")
        
        # Store number of real features for later use
        self._n_real_features = n_real_features
        
        # Compute outlier bounds ONLY on real features (not padded zeros)
        remove_outliers = self._get_config_value(self.preprocessing_config, 'remove_outliers', True)
        outlier_quantile = self._get_config_value(self.preprocessing_config, 'outlier_quantile', 0.99)
        
        if remove_outliers:
            lower_quantile = 1.0 - outlier_quantile
            upper_quantile = outlier_quantile
            # Compute bounds only on real features
            lower_bounds_real = np.quantile(X, lower_quantile, axis=0)
            upper_bounds_real = np.quantile(X, upper_quantile, axis=0)
            
            # Pad bounds to model size (padded features get bounds that won't affect zeros)
            if n_real_features < model_n_features:
                pad_size = model_n_features - n_real_features
                self._lower_bounds = np.concatenate([lower_bounds_real, np.zeros(pad_size)])
                self._upper_bounds = np.concatenate([upper_bounds_real, np.zeros(pad_size)])
            else:
                self._lower_bounds = lower_bounds_real
                self._upper_bounds = upper_bounds_real
            
            # Apply outlier removal for computing standardization stats
            X = np.clip(X, lower_bounds_real, upper_bounds_real)
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Computed outlier bounds at quantile {outlier_quantile}")
        
        # Compute standardization stats ONLY on real features (not padded zeros)
        feature_standardize = self._get_config_value(self.preprocessing_config, 'feature_standardize', True)
        
        if feature_standardize:
            # Compute stats only on real features
            X_mean_real = np.mean(X, axis=0, keepdims=True)
            X_std_real = np.std(X, axis=0, keepdims=True)
            X_std_real = np.where(X_std_real < 1e-8, 1.0, X_std_real)  # Avoid division by zero
            
            # Pad stats to model size (padded features get mean=0, std=1 so they stay as zeros)
            if n_real_features < model_n_features:
                pad_size = model_n_features - n_real_features
                self._X_mean = np.concatenate([X_mean_real, np.zeros((1, pad_size))], axis=1)
                self._X_std = np.concatenate([X_std_real, np.ones((1, pad_size))], axis=1)
            else:
                self._X_mean = X_mean_real
                self._X_std = X_std_real
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Computed standardization stats on {n_real_features} real features")
        
        # Compute target scaling stats
        if Y is not None:
            Y = np.asarray(Y, dtype=np.float32).flatten()
            self._y_min = float(np.min(Y))
            self._y_max = float(np.max(Y))
            self._y_range = max(self._y_max - self._y_min, 1e-8)
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Target range: [{self._y_min:.4f}, {self._y_max:.4f}]")
        
        self._fitted = True
        return self
    
    def _preprocess_features(
        self,
        X: np.ndarray,
        fit_mode: bool = False,
    ) -> np.ndarray:
        """
        Apply feature preprocessing (truncation/padding, outlier removal, standardization).
        
        The preprocessing stats (bounds, mean, std) were computed on real features only,
        with padded positions getting bounds=0, mean=0, std=1. This ensures:
        - Real features are properly preprocessed using their actual statistics
        - Padded features (zeros) remain as zeros after preprocessing: (0 - 0) / 1 = 0
        
        Args:
            X: Features, shape (N, L) or (B, N, L)
            fit_mode: If True, use the data itself for stats (for training data)
            
        Returns:
            Preprocessed features with shape matching model's num_features
        """
        X = np.asarray(X, dtype=np.float32)
        original_shape = X.shape
        is_batched = X.ndim == 3
        
        if is_batched:
            B, N, L = X.shape
        else:
            N, L = X.shape
            B = 1
            X = X[np.newaxis, ...]  # Add batch dim for uniform processing
        
        model_n_features = self.model.num_features
        
        # Truncate or pad features to model size
        # Note: The stored stats are already padded to model_n_features, so we can
        # safely pad here and apply the stats - padded features will stay as zeros
        if L > model_n_features:
            X = X[:, :, :model_n_features]
        elif L < model_n_features:
            pad_width = model_n_features - L
            X = np.concatenate([X, np.zeros((B, N, pad_width), dtype=np.float32)], axis=2)
        
        # Apply outlier removal using stored bounds
        # Padded positions have bounds=0, so zeros stay as zeros
        remove_outliers = self._get_config_value(self.preprocessing_config, 'remove_outliers', True)
        if remove_outliers and self._lower_bounds is not None and self._upper_bounds is not None:
            X = np.clip(X, self._lower_bounds, self._upper_bounds)
        
        # Apply standardization using stored stats
        # Padded positions have mean=0, std=1, so zeros stay as zeros: (0 - 0) / 1 = 0
        feature_standardize = self._get_config_value(self.preprocessing_config, 'feature_standardize', True)
        if feature_standardize and self._X_mean is not None and self._X_std is not None:
            X = (X - self._X_mean) / self._X_std
        
        # Remove batch dim if input wasn't batched
        if not is_batched:
            X = X[0]
        
        return X
    
    def _preprocess_targets(self, Y: np.ndarray) -> np.ndarray:
        """
        Scale targets to [-1, 1] range.
        
        Args:
            Y: Targets, shape (N,) or (N, 1) or (B, N) or (B, N, 1)
            
        Returns:
            Scaled targets in [-1, 1] range
        """
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim > 1 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        
        # Scale to [-1, 1]: Y_scaled = 2.0 * (Y - ymin) / range - 1.0
        Y_scaled = 2.0 * (Y - self._y_min) / self._y_range - 1.0
        return Y_scaled
    
    def _apply_test_feature_masking(self, X_test: np.ndarray) -> np.ndarray:
        """
        Apply random feature masking to test features.
        
        This matches the training-time augmentation in BasicProcessing where a 
        fraction of non-zero features in the test set are randomly masked (set to 0).
        The model is trained to be robust to this distribution shift.
        
        Args:
            X_test: Test features, shape (N, L) - should be preprocessed (standardized)
            
        Returns:
            X_test with a fraction of features masked (set to 0)
        """
        if self.test_feature_mask_fraction <= 0.0:
            return X_test
        
        X_test = X_test.copy()  # Don't modify original
        
        # Match BasicProcessing: identify non-zero columns (columns with at least one non-zero value)
        # Use a small epsilon to account for numerical precision
        eps = 1e-8
        column_is_nonzero = np.abs(X_test).sum(axis=0) > eps
        nonzero_cols = np.where(column_is_nonzero)[0].tolist()
        
        if len(nonzero_cols) == 0:
            # All columns are zero, nothing to mask
            return X_test
        
        # Calculate number of features to mask (matching BasicProcessing logic)
        n_to_mask = int(len(nonzero_cols) * self.test_feature_mask_fraction)
        
        if n_to_mask == 0:
            return X_test
        
        # Use random_state for reproducibility if set
        rng = np.random.RandomState(self.random_state)
        
        # Randomly select features to mask from non-zero columns
        cols_to_mask = rng.choice(nonzero_cols, size=n_to_mask, replace=False)
        
        # Mask the selected columns by setting them to zero
        X_test[:, cols_to_mask] = 0.0
        
        if self.verbose:
            print(f"[PreprocessingGraphConditionedPFN] Masked {n_to_mask}/{len(nonzero_cols)} non-zero features (fraction={self.test_feature_mask_fraction})")
        
        return X_test
    
    def _inverse_transform_predictions(self, Y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions from [-1, 1] back to original scale.
        
        Args:
            Y_scaled: Scaled predictions in [-1, 1] range
            
        Returns:
            Predictions in original scale
        """
        # Inverse: Y_original = (Y_scaled + 1.0) * range / 2.0 + ymin
        Y_original = (Y_scaled + 1.0) * self._y_range / 2.0 + self._y_min
        return Y_original
    
    def _inverse_transform_cate(self, cate_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform CATE predictions from scaled to original scale.
        
        For CATE (difference), we only need to scale by the range.
        
        Args:
            cate_scaled: Scaled CATE predictions
            
        Returns:
            CATE in original scale
        """
        # CATE_original = CATE_scaled * range / 2.0
        return cate_scaled * self._y_range / 2.0
    
    def _pad_or_truncate_samples(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: Optional[np.ndarray],
        max_samples: int,
        is_train: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
        """
        Pad or truncate samples to match model's expected sample count.
        
        Args:
            X: Features, shape (N, L)
            T: Treatment, shape (N,) or (N, 1)
            Y: Targets, shape (N,) or None
            max_samples: Maximum number of samples
            is_train: Whether this is training data (affects truncation strategy)
            
        Returns:
            Tuple of (X, T, Y, n_original) where n_original is the original sample count
        """
        n_samples = X.shape[0]
        n_original = n_samples
        
        # Ensure T is 2D
        T = np.asarray(T, dtype=np.float32)
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        
        # Truncate if needed
        if n_samples > max_samples:
            if is_train:
                # Random subsample for training
                indices = np.random.choice(n_samples, max_samples, replace=False)
                X = X[indices]
                T = T[indices]
                if Y is not None:
                    Y = Y[indices] if Y.ndim == 1 else Y[indices]
            else:
                # Just truncate for test (caller should handle batching)
                X = X[:max_samples]
                T = T[:max_samples]
                if Y is not None:
                    Y = Y[:max_samples] if Y.ndim == 1 else Y[:max_samples]
            n_samples = max_samples
        
        # Pad test samples if needed (model can handle variable-length train data, no padding needed)
        if not is_train and n_samples < max_samples:
            pad_size = max_samples - n_samples
            X = np.vstack([X, np.zeros((pad_size, X.shape[1]), dtype=np.float32)])
            T = np.vstack([T, np.zeros((pad_size, T.shape[1]), dtype=np.float32)])
            if Y is not None:
                Y = np.concatenate([Y, np.zeros(pad_size, dtype=np.float32)])
        
        return X, T, Y, n_original
    
    def _build_adjacency_matrix(
        self,
        n_real_features: int,
    ) -> np.ndarray:
        """
        Build a partial adjacency matrix for causal inference.
        
        Creates a matrix with known causal structure:
        - T -> Y (treatment causes outcome)
        - Real features -> T (features cause treatment)
        - Real features -> Y (features cause outcome)
        - Padded features have no edges (-1)
        
        Matrix ordering matches the dataset/model convention:
        - Position 0: Treatment (T)
        - Position 1: Outcome (Y)
        - Positions 2 to L+1: Feature variables (X[:,0] to X[:,L-1])
        
        Args:
            n_real_features: Number of real (non-padded) features
            
        Returns:
            Adjacency matrix of shape (model_n_features + 2, model_n_features + 2)
        """
        model_n_features = self.model.num_features
        
        # Initialize all as unknown (0)
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # Position mapping matching InterventionalDataset convention:
        # - Position 0: Treatment (T)
        # - Position 1: Outcome (Y)
        # - Positions 2 to L+1: Features
        T_idx = 0
        Y_idx = 1
        feature_offset = 2  # Features start at position 2
        
        # Known edges
        # 1. T -> Y (treatment causes outcome)
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # 2. Real features -> T (features cause treatment)
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
        
        # 3. Real features -> Y (features cause outcome)
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
        
        # 4. Padded features have no edges (-1)
        for i in range(n_real_features, model_n_features):
            feat_idx = feature_offset + i
            adjacency_matrix[feat_idx, :] = -1.0
            adjacency_matrix[:, feat_idx] = -1.0
            adjacency_matrix[feat_idx, feat_idx] = -1.0
        
        return adjacency_matrix
    
    def _hierarchical_cluster(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        Y_train: np.ndarray,
        max_n_train: int,
    ) -> np.ndarray:
        """
        Apply hierarchical clustering to keep cluster sizes <= max_n_train.
        
        Strategy:
        1. Initial k-means with k = N // max_n_train + 1
        2. For oversized clusters: split with k-means (k=2)
        3. For still-oversized sub-clusters: random split into equal parts
        
        Args:
            X_train: Training features (N, L)
            T_train: Training treatment (N,) or (N, 1)
            Y_train: Training targets (N,) or (N, 1)
            max_n_train: Maximum cluster size
            
        Returns:
            Cluster assignments for each training sample (N,)
        """
        N = X_train.shape[0]
        k_initial = N // max_n_train + 1
        
        if self.verbose:
            print(f"[PreprocessingGraphConditionedPFN] Clustering: Initial k-means with k={k_initial}")
        
        # Initial clustering
        rng = np.random.RandomState(self.random_state)
        kmeans = KMeans(n_clusters=k_initial, random_state=self.random_state, n_init=10)
        initial_labels = kmeans.fit_predict(X_train)
        
        # Track final cluster assignments (will be renumbered)
        cluster_assignments = np.copy(initial_labels)
        next_cluster_id = k_initial
        
        # Check each initial cluster
        for cluster_id in range(k_initial):
            cluster_mask = initial_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size <= max_n_train:
                continue
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN]   Cluster {cluster_id} has {cluster_size} samples (> {max_n_train}), splitting...")
            
            # Extract cluster data
            X_cluster = X_train[cluster_mask]
            
            # Try k-means split (k=2)
            kmeans_split = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            sub_labels = kmeans_split.fit_predict(X_cluster)
            
            # Check sub-cluster sizes
            sub_sizes = [np.sum(sub_labels == 0), np.sum(sub_labels == 1)]
            
            if max(sub_sizes) <= max_n_train:
                # K-means split was successful
                if self.verbose:
                    print(f"[PreprocessingGraphConditionedPFN]     K-means split successful: {sub_sizes[0]} + {sub_sizes[1]} samples")
                
                # Relabel: keep first sub-cluster with original ID, assign new ID to second
                cluster_indices = np.where(cluster_mask)[0]
                sub_0_indices = cluster_indices[sub_labels == 0]
                sub_1_indices = cluster_indices[sub_labels == 1]
                
                cluster_assignments[sub_0_indices] = cluster_id
                cluster_assignments[sub_1_indices] = next_cluster_id
                next_cluster_id += 1
            else:
                # K-means split failed, use random split
                n_subclusters = (cluster_size + max_n_train - 1) // max_n_train  # Ceiling division
                
                if self.verbose:
                    print(f"[PreprocessingGraphConditionedPFN]     K-means split failed, random splitting into {n_subclusters} sub-clusters")
                
                # Random permutation and split
                cluster_indices = np.where(cluster_mask)[0]
                shuffled_indices = rng.permutation(cluster_indices)
                
                # Assign to sub-clusters
                for i, idx in enumerate(shuffled_indices):
                    subcluster_id = cluster_id if i % n_subclusters == 0 else next_cluster_id + (i % n_subclusters) - 1
                    cluster_assignments[idx] = subcluster_id
                
                next_cluster_id += n_subclusters - 1
        
        # Renumber clusters to be consecutive starting from 0
        unique_labels = np.unique(cluster_assignments)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        cluster_assignments = np.array([label_map[label] for label in cluster_assignments])
        
        return cluster_assignments
    
    def _assign_to_clusters(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        cluster_assignments: np.ndarray,
    ) -> np.ndarray:
        """
        Assign test samples to nearest training cluster centroid.
        
        Computes the centroid of each training cluster and assigns each test sample
        to the cluster with the nearest centroid (using Euclidean distance).
        
        Args:
            X_test: Test features (M, L)
            X_train: Training features (N, L)
            cluster_assignments: Cluster labels for training samples (N,)
                Should be consecutive integers starting from 0
            
        Returns:
            Cluster assignments for test samples (M,) with same label space as cluster_assignments
        """
        # Compute cluster centroids
        unique_clusters = np.unique(cluster_assignments)
        centroids = np.zeros((len(unique_clusters), X_train.shape[1]))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_assignments == cluster_id
            centroids[i] = X_train[cluster_mask].mean(axis=0)
        
        # Assign each test sample to nearest centroid
        M = X_test.shape[0]
        test_assignments = np.zeros(M, dtype=int)
        
        for i in range(M):
            distances = np.linalg.norm(centroids - X_test[i], axis=1)
            nearest_cluster_idx = np.argmin(distances)
            test_assignments[i] = unique_clusters[nearest_cluster_idx]
        
        return test_assignments
    
    def _predict_single_cluster(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs_scaled: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        n_real_features: int,
        adjacency_matrix: Optional[np.ndarray],
        prediction_type: str,
        num_samples: int,
        max_n_test: int,
    ) -> np.ndarray:
        """
        Make predictions for a single cluster of data.
        
        Handles test batching internally if needed.
        """
        n_test_original = X_intv.shape[0]
        
        # Handle test samples in batches if needed
        if n_test_original > max_n_test:
            all_preds = []
            for batch_start in range(0, n_test_original, max_n_test):
                batch_end = min(batch_start + max_n_test, n_test_original)
                batch_size = batch_end - batch_start
                
                X_intv_batch = X_intv[batch_start:batch_end]
                T_intv_batch = T_intv[batch_start:batch_end]
                
                X_intv_batch, T_intv_batch, _, _ = self._pad_or_truncate_samples(
                    X_intv_batch, T_intv_batch, None, max_n_test, is_train=False
                )
                
                # Build adjacency matrix
                adj_matrix = adjacency_matrix if adjacency_matrix is not None else self._build_adjacency_matrix(n_real_features)
                
                # Call parent predict
                preds_batch = super().predict(
                    X_obs=X_obs,
                    T_obs=T_obs,
                    Y_obs=Y_obs_scaled,
                    X_intv=X_intv_batch,
                    T_intv=T_intv_batch,
                    adjacency_matrix=adj_matrix,
                    prediction_type=prediction_type,
                    num_samples=num_samples,
                    batched=False,
                )
                
                # Only keep non-padded predictions
                all_preds.append(preds_batch[:batch_size])
            
            return np.concatenate(all_preds)
        else:
            X_intv_padded, T_intv_padded, _, n_test = self._pad_or_truncate_samples(
                X_intv, T_intv, None, max_n_test, is_train=False
            )
            
            # Build adjacency matrix
            adj_matrix = adjacency_matrix if adjacency_matrix is not None else self._build_adjacency_matrix(n_real_features)
            
            # Call parent predict
            preds = super().predict(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs_scaled,
                X_intv=X_intv_padded,
                T_intv=T_intv_padded,
                adjacency_matrix=adj_matrix,
                prediction_type=prediction_type,
                num_samples=num_samples,
                batched=False,
            )
            
            # Only keep non-padded predictions
            return preds[:n_test_original]
    
    def predict(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        prediction_type: Literal["point", "mode", "mean", "sample"] = "mean",
        num_samples: int = 100,
        batched: bool = False,
        inverse_transform: bool = True,
    ) -> np.ndarray:
        """
        Make predictions with automatic preprocessing.
        
        This method:
        1. Preprocesses features (truncation/padding, outlier removal, standardization)
        2. Scales targets to [-1, 1]
        3. Pads/truncates samples to match model capacity
        4. Builds adjacency matrix if not provided
        5. Calls parent predict()
        6. Inverse transforms predictions to original scale
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational treatment (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional treatment (M,) or (M, 1)
            adjacency_matrix: Causal graph (L+2, L+2). If None, auto-constructed.
            prediction_type: "point", "mode", "mean", or "sample"
            num_samples: Number of samples for prediction_type="sample"
            batched: If True, expects batched inputs
            inverse_transform: If True, inverse transform predictions to original scale
            
        Returns:
            Predictions in original scale (if inverse_transform=True)
        """

        
        if not self._fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")
        
        if batched:
            raise NotImplementedError("Batched mode not yet implemented for PreprocessingGraphConditionedPFN")
        
        # Convert to numpy
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        
        n_real_features = min(X_obs.shape[1], self.model.num_features)
        
        # Preprocess features
        X_obs = self._preprocess_features(X_obs)
        X_intv = self._preprocess_features(X_intv)
        
        # Apply test feature masking to interventional (test) features
        # This matches the training-time augmentation
        X_intv = self._apply_test_feature_masking(X_intv)
        
        # Scale targets
        Y_obs_scaled = self._preprocess_targets(Y_obs)
        
        # Get max sample counts from config
        max_n_train = self._get_config_value(self.dataset_config, 'max_number_train_samples_per_dataset', X_obs.shape[0])
        max_n_test = self._get_config_value(self.dataset_config, 'max_number_test_samples_per_dataset', X_intv.shape[0])
        
        n_train_original = X_obs.shape[0]
        n_test_original = X_intv.shape[0]
        
        # Check if we need clustering (train samples exceed max_n_train)
        if self.use_clustering and n_train_original > max_n_train:
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Using clustering: {n_train_original} train samples > {max_n_train} max")
            
            # Apply hierarchical clustering
            cluster_assignments = self._hierarchical_cluster(X_obs, T_obs, Y_obs_scaled, max_n_train)
            n_clusters = len(np.unique(cluster_assignments))
            
            if self.verbose:
                unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
                print(f"[PreprocessingGraphConditionedPFN] Created {n_clusters} clusters, sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
            
            # Assign test samples to clusters
            test_cluster_assignments = self._assign_to_clusters(X_intv, X_obs, cluster_assignments)
            
            # Initialize predictions array
            preds = np.zeros(n_test_original, dtype=np.float32)
            
            # Process each cluster
            for cluster_id in np.unique(cluster_assignments):
                # Get training indices for this cluster
                train_mask = cluster_assignments == cluster_id
                
                # Get test indices for this cluster
                test_mask = test_cluster_assignments == cluster_id
                test_count = np.sum(test_mask)
                
                if test_count == 0:
                    continue
                
                # Extract cluster data
                X_obs_cluster = X_obs[train_mask]
                T_obs_cluster = T_obs[train_mask]
                Y_obs_cluster = Y_obs_scaled[train_mask]
                X_intv_cluster = X_intv[test_mask]
                T_intv_cluster = T_intv[test_mask]
                
                if self.verbose:
                    print(f"[PreprocessingGraphConditionedPFN]   Cluster {cluster_id}: {np.sum(train_mask)} train, {test_count} test samples")
                
                # Make predictions for this cluster
                cluster_preds = self._predict_single_cluster(
                    X_obs=X_obs_cluster,
                    T_obs=T_obs_cluster,
                    Y_obs_scaled=Y_obs_cluster,
                    X_intv=X_intv_cluster,
                    T_intv=T_intv_cluster,
                    n_real_features=n_real_features,
                    adjacency_matrix=adjacency_matrix,
                    prediction_type=prediction_type,
                    num_samples=num_samples,
                    max_n_test=max_n_test,
                )
                
                # Store predictions
                preds[test_mask] = cluster_preds
        else:
            # No clustering needed - use original logic with random subsampling if needed
            X_obs_proc, T_obs_proc, Y_obs_proc, n_train = self._pad_or_truncate_samples(
                X_obs, T_obs, Y_obs_scaled, max_n_train, is_train=True
            )
            
            # Make predictions using _predict_single_cluster (handles test batching)
            preds = self._predict_single_cluster(
                X_obs=X_obs_proc,
                T_obs=T_obs_proc,
                Y_obs_scaled=Y_obs_proc,
                X_intv=X_intv,
                T_intv=T_intv,
                n_real_features=n_real_features,
                adjacency_matrix=adjacency_matrix,
                prediction_type=prediction_type,
                num_samples=num_samples,
                max_n_test=max_n_test,
            )
        
        # Inverse transform
        if inverse_transform:
            preds = self._inverse_transform_predictions(preds)
        
        return preds
    
    def predict_cate(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        prediction_type: Literal["point", "mode", "mean"] = "mean",
    ) -> np.ndarray:
        """
        Predict Conditional Average Treatment Effect (CATE).
        
        CATE = E[Y|T=1, X] - E[Y|T=0, X]
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational treatment (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features to predict CATE for (M, L)
            adjacency_matrix: Causal graph (L+2, L+2). If None, auto-constructed.
            prediction_type: "point", "mode", or "mean"
            
        Returns:
            CATE predictions of shape (M,)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")
        
        n_test = X_intv.shape[0]
        
        # Predict Y for T=1
        T_intv_1 = np.ones((n_test, 1), dtype=np.float32)
        y_pred_1 = self.predict(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs,
            X_intv=X_intv,
            T_intv=T_intv_1,
            adjacency_matrix=adjacency_matrix,
            prediction_type=prediction_type,
            inverse_transform=False,  # Don't inverse transform yet
        )
        
        # Predict Y for T=0
        T_intv_0 = np.zeros((n_test, 1), dtype=np.float32)
        y_pred_0 = self.predict(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs,
            X_intv=X_intv,
            T_intv=T_intv_0,
            adjacency_matrix=adjacency_matrix,
            prediction_type=prediction_type,
            inverse_transform=False,  # Don't inverse transform yet
        )
        
        # CATE = E[Y|T=1,X] - E[Y|T=0,X] (in scaled space)
        cate_scaled = y_pred_1 - y_pred_0
        
        # Inverse transform CATE to original scale
        cate = self._inverse_transform_cate(cate_scaled)
        
        return cate
    
    def log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Compute log-likelihood with automatic preprocessing.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational treatment (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional treatment (M,) or (M, 1)
            Y_intv: Interventional targets (ground truth) (M,) or (M, 1)
            adjacency_matrix: Causal graph (L+2, L+2). If None, auto-constructed.
            batched: If True, expects batched inputs
            
        Returns:
            Log-likelihood values of shape (M,)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")
        
        if batched:
            raise NotImplementedError("Batched mode not yet implemented for PreprocessingGraphConditionedPFN")
        
        # Convert to numpy
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        Y_intv = np.asarray(Y_intv, dtype=np.float32)
        
        n_real_features = min(X_obs.shape[1], self.model.num_features)
        
        # Preprocess features
        X_obs = self._preprocess_features(X_obs)
        X_intv = self._preprocess_features(X_intv)
        
        # Apply test feature masking to interventional (test) features
        X_intv = self._apply_test_feature_masking(X_intv)
        
        # Scale targets
        Y_obs_scaled = self._preprocess_targets(Y_obs)
        Y_intv_scaled = self._preprocess_targets(Y_intv)
        
        # Get max sample counts from config
        max_n_train = self._get_config_value(self.dataset_config, 'max_number_train_samples_per_dataset', X_obs.shape[0])
        max_n_test = self._get_config_value(self.dataset_config, 'max_number_test_samples_per_dataset', X_intv.shape[0])
        
        # Pad/truncate samples
        X_obs, T_obs, Y_obs_scaled, n_train = self._pad_or_truncate_samples(
            X_obs, T_obs, Y_obs_scaled, max_n_train, is_train=True
        )
        
        n_test_original = X_intv.shape[0]
        X_intv, T_intv, Y_intv_scaled, n_test = self._pad_or_truncate_samples(
            X_intv, T_intv, Y_intv_scaled, max_n_test, is_train=False
        )
        
        # Build adjacency matrix
        if adjacency_matrix is None:
            adjacency_matrix = self._build_adjacency_matrix(n_real_features)
        
        # Call parent log_likelihood
        log_probs = super().log_likelihood(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs_scaled,
            X_intv=X_intv,
            T_intv=T_intv,
            Y_intv=Y_intv_scaled,
            adjacency_matrix=adjacency_matrix,
            batched=False,
        )
        
        # Only keep non-padded results
        return log_probs[:n_test_original]
    
    def predict_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        batched: bool = False,
    ) -> np.ndarray:
        """Alias for log_likelihood."""
        return self.log_likelihood(
            X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix, batched
        )
    
    def predict_negative_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        batched: bool = False,
    ) -> np.ndarray:
        """Compute negative log-likelihood."""
        return -self.log_likelihood(
            X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix, batched
        )


if __name__ == "__main__":
    """Test the PreprocessingGraphConditionedPFN class."""
    print("\n" + "="*80)
    print("TEST: PreprocessingGraphConditionedPFN")
    print("="*80)
    
    # Example usage (requires actual config and checkpoint)
    config_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
    checkpoint_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
    
    import os
    if os.path.exists(config_path) and os.path.exists(checkpoint_path):
        print(f"\nLoading model from:")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        wrapper = PreprocessingGraphConditionedPFN(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device="cpu",
            verbose=True,
        )
        wrapper.load()
        
        # Create synthetic data
        np.random.seed(42)
        N_train, N_test, L = 100, 20, 10
        
        X_train = np.random.randn(N_train, L).astype(np.float32)
        T_train = np.random.randint(0, 2, (N_train, 1)).astype(np.float32)
        Y_train = np.random.randn(N_train).astype(np.float32)
        
        X_test = np.random.randn(N_test, L).astype(np.float32)
        T_test = np.random.randint(0, 2, (N_test, 1)).astype(np.float32)
        Y_test = np.random.randn(N_test).astype(np.float32)
        
        print(f"\nFitting preprocessing on training data...")
        wrapper.fit(X_train, T_train, Y_train)
        
        print(f"\nMaking predictions...")
        preds = wrapper.predict(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
            T_intv=T_test,
            prediction_type="mean",
        )
        print(f"  Predictions shape: {preds.shape}")
        print(f"  Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
        
        print(f"\nPredicting CATE...")
        cate = wrapper.predict_cate(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
        )
        print(f"  CATE shape: {cate.shape}")
        print(f"  CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
        
        print(f"\nComputing log-likelihood...")
        log_probs = wrapper.log_likelihood(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
            T_intv=T_test,
            Y_intv=Y_test,
        )
        print(f"  Log-likelihood shape: {log_probs.shape}")
        print(f"  Mean log-likelihood: {log_probs.mean():.4f}")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
    else:
        print(f"\nSkipping tests - config or checkpoint not found:")
        print(f"  Config exists: {os.path.exists(config_path)}")
        print(f"  Checkpoint exists: {os.path.exists(checkpoint_path)}")
