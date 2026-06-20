"""
Variant Generator for toxicity expression robustness evaluation.

Constructs character-level perturbations of known toxic keywords.
These variants are used ONLY for testing (Variant Recall metric),
never added to training data.
"""

import random
import re
from typing import List


class VariantGenerator:
    """
    Generates adversarial text variants via:
      - Leet speak substitution
      - Space evasion
      - Symbol insertion
      - Homophone substitution (lightweight)
    """
    
    LEET_MAP = {
        'a': ['4', '@'],
        'e': ['3'],
        'i': ['1', '!'],
        'o': ['0'],
        's': ['5', '$'],
        't': ['7'],
        'g': ['9'],
        'b': ['8'],
    }
    
    HOMOPHONES = {
        'phuck': 'fuck',
        'azz': 'ass',
        'biatch': 'bitch',
        'd1ck': 'dick',
        'c0ck': 'cock',
    }
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def leet_substitute(self, text: str, prob: float = 0.3) -> str:
        """Replace characters with leet equivalents."""
        result = []
        for ch in text.lower():
            if ch in self.LEET_MAP and self.rng.random() < prob:
                result.append(self.rng.choice(self.LEET_MAP[ch]))
            else:
                result.append(ch)
        return "".join(result)
    
    def space_evasion(self, text: str) -> str:
        """Insert spaces between characters, e.g. 'idiot' -> 'i d i o t'."""
        return " ".join(text.lower())
    
    def symbol_insertion(self, text: str, symbols: str = ".*#-~") -> str:
        """Insert symbols between characters, e.g. 'idiot' -> 'i.d.i.o.t'."""
        sep = self.rng.choice(symbols)
        return sep.join(text.lower())
    
    def homophone_replace(self, text: str) -> str:
        """Replace known toxic words with homophone variants."""
        words = text.lower().split()
        replaced = []
        for w in words:
            # Check if any homophone is a substring
            found = False
            for homo, orig in self.HOMOPHONES.items():
                if orig in w:
                    w = w.replace(orig, homo)
                    found = True
                    break
            replaced.append(w)
        return " ".join(replaced)
    
    def generate_variant(self, text: str, mode: str = "random") -> str:
        """
        Generate a single variant of the input text.
        
        Args:
            text: Original text (expected to contain toxic keywords)
            mode: One of ['leet', 'space', 'symbol', 'homophone', 'random']
        """
        if mode == "random":
            mode = self.rng.choice(["leet", "space", "symbol", "homophone"])
        
        if mode == "leet":
            return self.leet_substitute(text, prob=0.4)
        elif mode == "space":
            # Apply space evasion to each word independently with some prob
            words = text.split()
            new_words = []
            for w in words:
                if self.rng.random() < 0.5 and len(w) > 2:
                    new_words.append(self.space_evasion(w))
                else:
                    new_words.append(w)
            return " ".join(new_words)
        elif mode == "symbol":
            words = text.split()
            new_words = []
            for w in words:
                if self.rng.random() < 0.5 and len(w) > 2:
                    new_words.append(self.symbol_insertion(w))
                else:
                    new_words.append(w)
            return " ".join(new_words)
        elif mode == "homophone":
            return self.homophone_replace(text)
        else:
            return text
    
    def generate_batch(self, texts: List[str], n_variants: int = 1) -> List[str]:
        """Generate n variants per text."""
        results = []
        for text in texts:
            for _ in range(n_variants):
                results.append(self.generate_variant(text))
        return results
    
    @classmethod
    def extract_toxic_vocab(cls, texts: List[str], min_freq: int = 3) -> List[str]:
        """
        Extract frequent tokens from toxic texts (simple heuristic).
        In practice, this should use a curated toxic word list.
        """
        from collections import Counter
        words = []
        for t in texts:
            words.extend(re.findall(r'\b[a-z]{4,}\b', t.lower()))
        freq = Counter(words)
        # Return words that appear frequently (naive heuristic)
        return [w for w, c in freq.most_common(100) if c >= min_freq]
