"""
Toxicity Expression Structure-Aware Positional Encoding (Section 3.4 of problemDef).

PE_i = PE_abs(i) + W_q * q_i + W_m * m_i + W_v * v_i

where:
  q_i: emphasis intensity (punctuation density, repeated punctuation)
  m_i: markup/caps (continuous caps, *asterisks*, _underscores_)
  l_i: syntactic pattern (sentence length/structure)
  v_i: character variation (leet speak, space evasion)

Injected after embedding layer, before Transformer blocks.
Features are computed per-token (character position ranges),
NOT globally replicated — each token gets its own structural signals.
"""

import re
import torch
import torch.nn as nn


class ToxicAwarePE(nn.Module):
    """
    Computes additive toxicity-aware position/structure embeddings.
    
    Args:
        hidden_size: Transformer hidden dimension
        max_length: Maximum sequence length
        q_dim, m_dim, l_dim, v_dim: Output dims for each feature (default 1)
            q: emphasis intensity, m: markup/caps, l: syntactic pattern, v: char variation
    """
    
    # Correct leet character set: {4, 0, 1, 3, 5, 7, 8, 9} (no literal | or $)
    LEET_PATTERN = re.compile(r'[40135789]')
    # Repeated punctuation: sequences like "!!" or "??" or "?!?"
    EMPHASIS_PATTERN = re.compile(r'[!?]+')
    # Continuous caps: 3+ consecutive uppercase letters
    CAPS_PATTERN = re.compile(r'[A-Z]{3,}')
    # Markup emphasis: *text* or _text_
    ASTERISK_PATTERN = re.compile(r'\*[^*]+\*')
    UNDERSCORE_PATTERN = re.compile(r'_[^_]+_')
    # Space evasion: single characters separated by spaces
    SPACE_EVASION_PATTERN = re.compile(r'(?<=\s)\S(?=\s)')
    
    def __init__(
        self,
        hidden_size: int = 768,
        max_length: int = 128,
        q_dim: int = 1,
        m_dim: int = 1,
        l_dim: int = 1,
        v_dim: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        total_feat_dim = q_dim + m_dim + l_dim + v_dim
        
        # Projection from feature dim to hidden_size
        self.feat_proj = nn.Linear(total_feat_dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.feat_proj.weight, gain=0.01)
    
    def extract_surface_features(self, texts: list) -> torch.Tensor:
        """
        Extract per-token surface features for a batch of texts.
        Each token position receives features computed from its character range,
        NOT globally replicated — so "HATE" tokens get distinct signals from "I" tokens.
        
        Args:
            texts: List of raw text strings (batch_size,)
        
        Returns:
            feats: Tensor of shape [batch_size, max_length, 4] (q, m, l, v)
        """
        batch_size = len(texts)
        feats = torch.zeros(batch_size, self.max_length, 4)
        
        for b, text in enumerate(texts):
            text_lower = text.lower()
            
            # --- Build per-token character span map ---
            # Split on whitespace to get token boundaries
            words = text.split()
            # Reconstruct each token's (char_start, char_end) in the original string
            token_spans = []
            char_pos = 0
            for w in words:
                # Find this word in the remaining text, skipping leading whitespace
                while char_pos < len(text) and text[char_pos].isspace():
                    char_pos += 1
                start = char_pos
                end = char_pos + len(w)
                token_spans.append((start, end))
                char_pos = end
            
            n_tokens = min(len(token_spans), self.max_length)
            
            # --- Global text-level signal (l_i: length structure) ---
            word_count = len(words)
            l_val = min(word_count / 50.0, 1.0)
            
            # --- Pre-locate special matches with character positions ---
            # Emphasis: find all !? spans with their positions
            emphasis_spans = []
            for m in re.finditer(r'([!?])', text):
                emphasis_spans.append((m.start(), m.end()))
            emphasis_positions = set()
            for s, e in emphasis_spans:
                emphasis_positions.update(range(s, e))
            
            # Caps: find 3+ uppercase spans
            caps_spans = []
            for m in self.CAPS_PATTERN.finditer(text):
                caps_spans.append((m.start(), m.end()))
            caps_positions = set()
            for s, e in caps_spans:
                caps_positions.update(range(s, e))
            
            # Markup: *word* and _word_
            markup_positions = set()
            for m in self.ASTERISK_PATTERN.finditer(text):
                markup_positions.update(range(m.start(), m.end()))
            for m in self.UNDERSCORE_PATTERN.finditer(text):
                markup_positions.update(range(m.start(), m.end()))
            
            # Leet: digit-for-letter substitutions
            leet_positions = set()
            for m in self.LEET_PATTERN.finditer(text):
                leet_positions.add(m.start())
            
            # Space evasion: single chars between spaces
            evasion_positions = set()
            for m in self.SPACE_EVASION_PATTERN.finditer(text):
                evasion_positions.add(m.start())
            
            # --- Per-token feature computation ---
            for i in range(n_tokens):
                start, end = token_spans[i]
                token_chars = set(range(start, end))
                
                # q_i: emphasis intensity — does this token overlap with !? chars
                q_overlap = len(token_chars & emphasis_positions)
                q_val = min(q_overlap / max(end - start, 1), 1.0)
                
                # m_i: markup/caps — does this token overlap with caps or markup regions
                m_overlap = len(token_chars & caps_positions) + len(token_chars & markup_positions)
                m_val = min(m_overlap / max(end - start, 1), 1.0)
                
                # v_i: character variation — does this token contain leet or evasion chars
                v_overlap = len(token_chars & leet_positions) + len(token_chars & evasion_positions)
                v_val = min(v_overlap / max(end - start, 1.0), 1.0)
                
                feats[b, i, 0] = q_val
                feats[b, i, 1] = m_val
                feats[b, i, 2] = l_val  # l is global, but structurally meaningful per token
                feats[b, i, 3] = v_val
        
        return feats
    
    def forward(self, texts: list, base_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            texts: List of raw text strings (batch_size,)
            base_embeddings: Optional base token embeddings [B, L, H]
        
        Returns:
            additive_pe: Tensor of shape [B, L, H] to be added to embeddings.
        """
        device = base_embeddings.device if base_embeddings is not None else torch.device("cpu")
        
        feats = self.extract_surface_features(texts).to(device)
        # feats: [B, L, 4]
        
        additive = self.feat_proj(feats)  # [B, L, H]
        return additive
