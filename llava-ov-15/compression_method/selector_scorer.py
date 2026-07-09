import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


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
    Architecture: one LLM-style decoder layer (bidirectional attention, GQA, no RoPE) -> LayerNorm -> Linear.
    Architecture matches LLM decoder layer to allow weight copying from a pretrained LLM layer.
    The final projection is initialized close to zero to minimally interfere with the original attention_sum.
    """
    def __init__(self, in_features: int, num_kv_heads: int = 2,
                 intermediate_size: int = 11008, attention_bias: bool = True,
                 head_dim: int = 128, init_scale: float = 0.0001):
        super().__init__()
        self.in_features = in_features
        self.head_dim = head_dim
        self.num_heads = in_features // head_dim
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_heads // num_kv_heads
        self.scaling = self.head_dim ** -0.5

        # --- Bidirectional self-attention (GQA, no causal mask, no RoPE) ---
        self.q_proj = nn.Linear(in_features, self.num_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(in_features, num_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(in_features, num_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * head_dim, in_features, bias=False)

        # --- MLP (SwiGLU) ---
        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, in_features, bias=False)

        # --- LayerNorms ---
        self.input_layernorm = RMSNorm(in_features)
        self.post_attention_layernorm = RMSNorm(in_features)
        self.final_layernorm = nn.LayerNorm(in_features)

        # --- Score projection (multi-layer MLP) ---
        embed_dim = in_features
        self.score_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, embed_dim // 8, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim // 8, embed_dim // 16, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim // 16, 1, bias=True),
        )

        # Initialize last layer to near-zero for minimal initial interference
        nn.init.normal_(self.score_proj[-1].weight, std=init_scale)
        nn.init.zeros_(self.score_proj[-1].bias)

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

        # --- Self-attention (bidirectional, GQA, no causal mask) ---
        residual = x
        x = self.input_layernorm(x)

        query_states = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Flash attention (is_causal=False for bidirectional)
        query_states = query_states.transpose(1, 2)  # [B, N, H, D]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if is_flash_attn_2_available():
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                query_length=seq_len,
                is_causal=False,
                dropout=0.0,
            )
        else:
            # Fallback: manual attention with GQA
            from torch.nn.functional import scaled_dot_product_attention as sdpa
            attn_output = sdpa(
                query_states, key_states, value_states,
                dropout_p=0.0, is_causal=False,
            )

        attn_output = attn_output.view(bsz, seq_len, -1)
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
