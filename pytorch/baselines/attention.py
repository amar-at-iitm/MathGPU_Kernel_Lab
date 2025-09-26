"""
PyTorch Scaled Dot-Product Multi-Head Attention Baseline.
Mathematical formulation:
    S = (Q @ K.T) / sqrt(d_k)
    P = softmax(S, dim=-1)
    O = P @ V
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def scaled_dot_product_attention_pytorch(q: torch.Tensor,
                                        k: torch.Tensor,
                                        v: torch.Tensor,
                                        is_causal: bool = False) -> torch.Tensor:
    r"""
    Reference Scaled Dot-Product Attention:
    q, k, v: (B, H, N, D) or (B, N, D)
    returns: (B, H, N, D) or (B, N, D)
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale # (B, H, N, N)

    if is_causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
        scores = scores + mask

    probs = F.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    return output
