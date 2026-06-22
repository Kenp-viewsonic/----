"""
Toxic Semantic Anchor Prefix (Section 3.2 of problemDef).

Prefix-Tuning variant where prefix embeddings are initialized via
K-means clustering of base-class [CLS] representations, rather than
random initialization.

At stage k:
    P_k = alpha * P_proto + (1 - alpha) * Theta_P[k]

Injected into K and V (not Q) at every transformer layer.
"""

import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class ToxicSemanticPrefix(nn.Module):
    """
    Args:
        hidden_size: Transformer hidden dimension (768 for roberta-base)
        num_layers: Number of transformer layers to inject prefix into
        prefix_length: Number of prefix tokens (m)
        n_anchors: Number of K-means clusters for initialization
        alpha: Blend weight between proto anchor and learned residual
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        prefix_length: int = 10,
        n_anchors: int = 5,
        alpha: float = 0.7,
        stage_alpha: dict = None,
        layerwise_alpha: list = None,
        init_random: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prefix_length = prefix_length
        self.n_anchors = n_anchors
        self.alpha = alpha
        self.stage_alpha = {str(k): float(v) for k, v in (stage_alpha or {}).items()}
        # Default: stage1 alpha=0.3, stage2 alpha=0.6; base uses construction-time alpha.
        if not self.stage_alpha:
            self.stage_alpha = {"1": 0.3, "2": 0.6}
        self.layerwise_alpha = list(layerwise_alpha) if layerwise_alpha is not None else None
        
        # Flag to indicate if proto has been initialized
        self._proto_initialized = False
        
        # Proto anchor: initialized from K-means, then frozen or low-lr
        self.register_buffer(
            "P_proto",
            torch.zeros(num_layers, prefix_length, hidden_size)
        )
        
        if init_random:
            nn.init.xavier_uniform_(self.P_proto)
            self._proto_initialized = True
        
        # Stage-specific learnable residual
        # We keep a dict of residuals per stage to avoid losing history
        self.stage_residuals = nn.ParameterDict()
        self.current_stage = None
    
    def init_from_kmeans(self, cls_embeddings: torch.Tensor):
        """
        Initialize P_proto from K-means clustering of base-class [CLS] embeddings.
        
        When prefix_length > n_anchors (e.g. 10 > 5), the extra positions are
        filled with a LEARNABLE embedding (nn.Parameter) initialized via Xavier,
        NOT by cyclically repeating the centroids. This avoids redundant
        information and gives the model freedom to learn complementary anchors.
        
        Args:
            cls_embeddings: Tensor of shape [N, hidden_size] from base-class toxic samples.
        """
        if cls_embeddings.dim() == 1:
            cls_embeddings = cls_embeddings.unsqueeze(0)
        
        N = cls_embeddings.shape[0]
        n_clusters = min(self.n_anchors, N)
        
        # K-means on CPU (sklearn)
        embs_np = cls_embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embs_np)
        
        # Centroids shape: [n_clusters, hidden_size]
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        
        # Fill prefix_length with centroids + learned padding (no cyclic repeat)
        if self.prefix_length <= n_clusters:
            proto = centroids[:self.prefix_length]
        else:
            # Use centroids for the first n_clusters positions
            extra = self.prefix_length - n_clusters
            # Learnable extra positions (shared across layers for parameter efficiency)
            if not hasattr(self, '_proto_padding'):
                # Must be on same device as the rest of the model (init_from_kmeans
                # is called AFTER model.to(device), so move explicitly)
                device = self.P_proto.device
                self._proto_padding = nn.Parameter(
                    torch.zeros(1, extra, self.hidden_size, device=device)
                )
                nn.init.xavier_uniform_(self._proto_padding)
            # Concatenate centroids + learned padding
            proto = torch.cat([
                centroids,
                self._proto_padding.squeeze(0).to(centroids.device)
            ], dim=0)
        
        # Same proto for all layers
        self.P_proto.copy_(proto.unsqueeze(0).expand(self.num_layers, -1, -1))
        self._proto_initialized = True
        self._kmeans_labels_ = kmeans.labels_
        self._kmeans_centroids_ = centroids.clone()
        print(f"[Prefix] Initialized {n_clusters} K-means centroids + {self.prefix_length - n_clusters} learned padding positions.")
    
    def add_stage(self, stage_idx: int):
        """Register a new learnable residual for the given stage."""
        key = str(stage_idx)
        if key not in self.stage_residuals:
            self.stage_residuals[key] = nn.Parameter(
                torch.zeros(self.num_layers, self.prefix_length, self.hidden_size)
            )
            # Xavier init for residual
            nn.init.xavier_uniform_(self.stage_residuals[key])
        self.current_stage = stage_idx
    
    def get_prefix(self, stage_idx: int = None):
        """
        Compute P_k for a given stage.
        Returns tensor of shape [num_layers, prefix_length, hidden_size].
        """
        if stage_idx is None:
            stage_idx = self.current_stage
        
        if stage_idx is None:
            # No stage set yet: return proto only
            return self.P_proto
        
        key = str(stage_idx)
        if key not in self.stage_residuals:
            # Fallback to proto if residual not exists
            return self.P_proto
        
        residual = self.stage_residuals[key]
        alpha = float(self.stage_alpha.get(key, self.alpha))
        if self.layerwise_alpha is None:
            P_k = alpha * self.P_proto + (1.0 - alpha) * residual
        else:
            layer_alpha = torch.tensor(
                self.layerwise_alpha,
                device=self.P_proto.device,
                dtype=self.P_proto.dtype,
            ).view(self.num_layers, 1, 1)
            P_k = layer_alpha * self.P_proto + (1.0 - layer_alpha) * residual
        return P_k
    
    def forward(self, stage_idx: int = None):
        """Return prefix embeddings for the given stage."""
        return self.get_prefix(stage_idx)
    
    def get_prefix_for_layer(self, layer_idx: int, stage_idx: int = None):
        """Get prefix for a specific layer. Shape: [prefix_length, hidden_size]."""
        P = self.get_prefix(stage_idx)
        return P[layer_idx]
