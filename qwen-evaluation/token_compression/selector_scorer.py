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
    Transformer Scorer using a single bidirectional transformer layer for importance scoring.
    Architecture: one LLM-style decoder layer (bidirectional attention, no RoPE) -> LayerNorm -> Linear.
    The final projection is initialized close to zero to minimally interfere with the original attention_sum.
    """
    def __init__(self, in_features: int, hidden_dim: int = 1792, init_scale: float = 0.0001):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # --- Bidirectional self-attention (no causal mask, no RoPE) ---
        self.head_dim = 128
        self.num_heads = in_features // self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_features, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(in_features, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(in_features, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, in_features, bias=False)

        # --- MLP (SwiGLU) ---
        self.gate_proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, in_features, bias=False)

        # --- LayerNorms ---
        self.input_layernorm = RMSNorm(in_features)
        self.post_attention_layernorm = RMSNorm(in_features)
        self.final_layernorm = nn.LayerNorm(in_features)

        # --- Score projection ---
        self.score_proj = nn.Linear(in_features, 1, bias=False)

        # Initialize final projection to near-zero for minimal initial interference
        nn.init.normal_(self.score_proj.weight, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates importance scores via a bidirectional transformer layer.

        Args:
            x (torch.Tensor): Visual tokens of shape [B, N, D]
                             (B: batch size, N: token count, D: embedding dim)
        Returns:
            torch.Tensor: Learned importance scores, shape [B, N]
        """
        bsz, seq_len, _ = x.shape

        # --- Self-attention (bidirectional, no causal mask) ---
        residual = x
        x = self.input_layernorm(x)

        query_states = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        x = residual + attn_output

        # --- MLP (SwiGLU) ---
        residual = x
        x = self.post_attention_layernorm(x)
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = residual + x

        # --- Final LayerNorm + Linear ---
        x = self.final_layernorm(x)
        scores = self.score_proj(x).squeeze(-1)  # [B, N]

        return scores


if __name__ == '__main__':
    model = TransformerScorer(in_features=3584)
    x = torch.randn(1, 220, 3584)
    y = model(x)
    print(y.shape)
    print(y)
