"""
Main model: RoBERTa-base + ToxicSemanticPrefix + DualBranchLoRA + ToxicAwarePE + HierarchicalRejectionGate.

Integrates all modules and exposes forward modes for semantic consolidation.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

from .toxic_prefix import ToxicSemanticPrefix
from .dual_lora import DualBranchLoRALayer
from .toxic_pe import ToxicAwarePE
from .rejection_gate import HierarchicalRejectionGate


class RobertaToxicClassifier(nn.Module):
    """
    FSCIL Toxic Comment Classifier with all proposed modules.
    
    Args:
        num_classes: Number of fine-grained toxic classes
        model_name: HuggingFace model name
        prefix_cfg: Dict with prefix hyperparameters
        lora_cfg: Dict with LoRA hyperparameters
        pe_cfg: Dict with ToxicAwarePE hyperparameters
        gate_cfg: Dict with rejection gate hyperparameters
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = "roberta-base",
        prefix_cfg: dict = None,
        lora_cfg: dict = None,
        pe_cfg: dict = None,
        gate_cfg: dict = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load base RoBERTa
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        # --- Prefix module ---
        pcfg = prefix_cfg or {}
        self.prefix_module = ToxicSemanticPrefix(
            hidden_size=self.hidden_size,
            num_layers=self.config.num_hidden_layers,
            prefix_length=pcfg.get("prefix_length", 10),
            n_anchors=pcfg.get("n_anchors", 5),
            alpha=pcfg.get("alpha", 0.7),
            init_random=pcfg.get("init_random", False),
        )
        self.prefix_length = self.prefix_module.prefix_length
        
        # --- ToxicAwarePE ---
        pecfg = pe_cfg or {}
        self.toxic_pe = ToxicAwarePE(
            hidden_size=self.hidden_size,
            max_length=pecfg.get("max_length", 128),
            q_dim=pecfg.get("q_dim", 1),
            m_dim=pecfg.get("m_dim", 1),
            l_dim=pecfg.get("l_dim", 1),
            v_dim=pecfg.get("v_dim", 1),
        ) if pecfg.get("enable", True) else None
        
        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )
        
        # --- Rejection Gate ---
        gcfg = gate_cfg or {}
        self.rejection_gate = HierarchicalRejectionGate(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            theta_coarse=gcfg.get("theta_coarse", 0.5),
            theta_fine=gcfg.get("theta_fine", 0.3),
            a=gcfg.get("a", 1.0),
            b=gcfg.get("b", 1.0),
            c=gcfg.get("c", 1.0),
            d=gcfg.get("d", 1.0),
            learnable_weights=gcfg.get("learnable_weights", False),
        )
        
        # Replace each layer's self-attention forward with prefix-aware version
        self._replace_attention_forwards()
        
        # Stage tracking
        self.current_stage = 0
        self._forward_mode = "full"
    
    def _replace_attention_forwards(self):
        """Bind prefix-injected forward to each RobertaSelfAttention layer."""
        for layer_idx, layer in enumerate(self.roberta.encoder.layer):
            attn = layer.attention.self
            # Bind layer-specific prefix injection
            attn.forward = self._make_prefix_attention_forward(attn, layer_idx)
    
    def _make_prefix_attention_forward(self, self_attn, layer_idx):
        """Create a new forward method for a specific attention layer."""
        original_forward = self_attn.forward
        classifier_ref = self
        idx = layer_idx
        
        def prefix_attention_forward(
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            **kwargs,
        ):
            # Only apply prefix injection when not using past_key_values (generation not supported)
            if past_key_values is not None or kwargs.get("encoder_hidden_states") is not None:
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            
            # Get prefix for this layer and stage
            prefix = classifier_ref.prefix_module.get_prefix_for_layer(idx, classifier_ref.current_stage)
            batch_size = hidden_states.size(0)
            prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)  # [B, m, H]
            
            # Projection helper (passes forward mode into DualBranchLoRA)
            def proj(module, x):
                if isinstance(module, DualBranchLoRALayer):
                    return module(x, mode=classifier_ref._forward_mode)
                return module(x)
            
            # Query: only on hidden_states (no prefix)
            Q = proj(self_attn.query, hidden_states)
            
            # Key and Value: prefix + hidden_states
            kv_input = torch.cat([prefix, hidden_states], dim=1)
            K = proj(self_attn.key, kv_input)
            V = proj(self_attn.value, kv_input)
            
            # Reshape for multi-head attention
            num_heads = self_attn.num_attention_heads
            head_dim = self_attn.attention_head_size
            
            Q = Q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self_attn.dropout(attn_probs)
            
            head_mask = kwargs.get("head_mask")
            if head_mask is not None:
                attn_probs = attn_probs * head_mask
            
            attn_output = torch.matmul(attn_probs, V)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, -1, num_heads * head_dim)
            
            # Return same structure as original RobertaSelfAttention.forward:
            # (context_layer, attn_probs, past_key_values)
            outputs = (attn_output,)
            if kwargs.get("output_attentions"):
                outputs += (attn_probs,)
            outputs += (past_key_values,)
            return outputs
        
        return prefix_attention_forward
    
    def set_stage(self, stage_idx: int):
        """Set current FSCIL stage."""
        self.current_stage = stage_idx
        self.prefix_module.add_stage(stage_idx)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        texts=None,
        labels=None,
        mode: str = "full",
        return_rejection: bool = True,
    ):
        """
        Args:
            input_ids, attention_mask: Standard transformer inputs
            texts: Raw text strings (needed for ToxicAwarePE and rejection gate)
            labels: Ground-truth labels [B, C]
            mode: 'full', 'stable_only', or 'base_only' (for semantic consolidation eval)
            return_rejection: If True, also compute rejection gate outputs
        
        Returns:
            dict with 'logits', 'cls_hidden', optionally 'rejection', 'loss'
        """
        self._forward_mode = mode
        
        # 1. Base RoBERTa embeddings
        embedding_output = self.roberta.embeddings(input_ids=input_ids)
        
        # 2. Add ToxicAwarePE if enabled
        if self.toxic_pe is not None and texts is not None:
            pe_additive = self.toxic_pe(texts, base_embeddings=embedding_output)
            embedding_output = embedding_output + pe_additive
        
        # 3. Pass through encoder (prefix injection is handled in replaced attention forwards)
        # Need to extend attention_mask to account for prefix tokens
        extended_attention_mask = None
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            prefix_mask = torch.ones(batch_size, self.prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
            extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            # get_extended_attention_mask converts 0/1 mask to float broadcasted mask for encoder
            extended_attention_mask = self.roberta.get_extended_attention_mask(
                extended_mask, extended_mask.shape
            )
        
        encoder_outputs = self.roberta.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        hidden_states = encoder_outputs[0]  # [B, seq_len, H]
        
        # 4. [CLS] pooling (prefix is only injected into K/V, so sequence length unchanged)
        cls_hidden = hidden_states[:, 0, :]  # [B, H]
        
        # 5. Classification head
        logits = self.classifier(cls_hidden)  # [B, C]
        
        output = {
            "logits": logits,
            "cls_hidden": cls_hidden,
        }
        
        # 6. Rejection gate
        if return_rejection:
            rejection = self.rejection_gate(cls_hidden, logits, texts=texts)
            output["rejection"] = rejection
        
        # 7. Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # Per-element loss: [B, C]
            element_loss = loss_fct(logits, labels.float())
            # Only average over positions where labels >= 0 (handles -1 ignore markers)
            # In ActiveLabelDataset path, all labels are 0/1 so mask is all-ones
            mask = (labels >= 0).float()
            active_count = mask.sum() + 1e-8
            bce_loss = (element_loss * mask).sum() / active_count
            output["loss"] = bce_loss
        
        return output
    
    def get_stable_cls_embedding(self, input_ids, attention_mask, texts=None):
        """Get [CLS] using only stable branch (for semantic consolidation delta_k)."""
        with torch.no_grad():
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                texts=texts,
                mode="stable_only",
                return_rejection=False,
            )
        return out["cls_hidden"]
    
    def expand_classifier(self, new_num_classes: int):
        """Expand classification head to include new classes."""
        old_head = self.classifier[1]
        old_num_classes = old_head.out_features
        
        if new_num_classes <= old_num_classes:
            return
        
        new_head = nn.Linear(self.hidden_size, new_num_classes)
        with torch.no_grad():
            new_head.weight[:old_num_classes, :] = old_head.weight
            new_head.bias[:old_num_classes] = old_head.bias
            # New classes are randomly initialized (will be trained in next stage)
        
        self.classifier[1] = new_head
        self.num_classes = new_num_classes
        self.rejection_gate.num_classes = new_num_classes
        print(f"[Classifier] Expanded from {old_num_classes} to {new_num_classes} classes.")
    
    def consolidate_plastic(self, merge: bool = True):
        """
        Trigger semantic consolidation for all DualBranchLoRALayer modules.
        If merge=True and delta_k < tau (evaluated externally), merge plastic to stable.
        If merge=False, freeze plastic as historical patch.
        """
        for name, module in self.named_modules():
            if isinstance(module, DualBranchLoRALayer):
                if merge:
                    module.merge_plastic_to_stable()
                else:
                    module.freeze_current_plastic()
