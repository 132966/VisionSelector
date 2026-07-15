import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TransformerScorer(nn.Module):
    """
    Simplified Transformer Scorer using bidirectional self-attention for importance scoring.
    Architecture: RMSNorm -> q/k_proj (GQA) -> attention weights -> mean.
    Only computes attention weights (QK^T), does not compute full attention output (no V, no o_proj).
    Weights are initialized from a pretrained LLM middle layer (q_proj, k_proj, input_layernorm).
    Attention is bidirectional (no causal mask, no RoPE).
    """
    def __init__(self, in_features: int, num_kv_heads: int = 2,
                 intermediate_size: int = 11008, attention_bias: bool = True,
                 head_dim: int = 128, init_scale: float = 0.0001,
                 init_budget: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.head_dim = head_dim
        self.num_heads = in_features // head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5

        # --- RMSNorm (for LLM weight loading) ---
        self.input_layernorm = RMSNorm(in_features)

        # --- Bidirectional self-attention (GQA, no causal mask, no RoPE) ---
        self.q_proj = nn.Linear(in_features, self.num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(in_features, num_kv_heads * head_dim, bias=attention_bias)

        # --- Learnable budget parameter ---
        self.budget = nn.Parameter(torch.tensor(init_budget, dtype=torch.float32))

    @property
    def budget_value(self):
        """Return budget clamped to (0.01, 0.99) for numerical stability."""
        return torch.clamp(self.budget, min=0.01, max=0.99)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates importance scores via simplified bidirectional self-attention.

        Args:
            x (torch.Tensor): Visual tokens of shape [B, N, D]
                             (B: batch size, N: token count, D: embedding dim)
        Returns:
            torch.Tensor: Learned importance scores, shape [B, N]
        """
        bsz, seq_len, _ = x.shape

        # --- RMSNorm ---
        x = self.input_layernorm(x)

        # --- Q and K projections (GQA) ---
        query_states = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        key_states = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, N, D]

        # Repeat K for GQA
        key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)  # [B, H, N, D]

        # --- Compute attention weights (bidirectional, no causal mask, no softmax) ---
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling  # [B, H, N, N]

        # --- Scores: average over heads and key dimension ---
        scores = attn_weights.mean(dim=(1, -1))  # [B, N]

        return scores


if __name__ == '__main__':
    model = TransformerScorer(in_features=3584)
    x = torch.randn(1, 220, 3584)
    y = model(x)
    print(y.shape)
    print(y)
