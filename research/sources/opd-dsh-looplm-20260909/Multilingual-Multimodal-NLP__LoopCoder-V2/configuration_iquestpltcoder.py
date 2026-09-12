"""IQuestPLTCoder model configuration.

Extends the IQuestCoder configuration with PLT (Parallel Loop Transformer)
specific parameters. PLT reuses the same physical transformer layers across
multiple loops, with cross-loop processing (CLP) and mixed attention (global
full-attention + local sliding-window attention gated per head) in loop 1+.

Reference: https://arxiv.org/abs/2510.24824
"""

from typing import Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class IQuestPLTCoderConfig(PretrainedConfig):
    r"""
    Configuration class for [`IQuestPLTCoderModel`].

    This is a PLT (Parallel Loop Transformer) variant of IQuestCoder. The model
    has `num_hidden_layers` physical transformer layers that are executed
    `plt_num_loops` times. Weights are shared across loops; each loop adds
    cross-loop processing and mixed attention via a learned per-head gate.

    Args:
        vocab_size (`int`, *optional*, defaults to 75904):
            Vocabulary size of the model (padded to be divisible by 128).
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 27648):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 14):
            Number of physical transformer layers (shared across all loops).
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for Grouped Query Attention (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of each attention head.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder (SwiGLU uses SiLU).
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int` or `list`, *optional*, defaults to `[2, 75864, 75869]`):
            End of stream token id(s).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input embedding and output projection weights.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE
            embeddings. Supports "linear", "dynamic", "yarn", "longrope", "llama3".
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the Q, K, V and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP gate/up/down projection layers.
        plt_num_loops (`int`, *optional*, defaults to 2):
            Number of times the physical transformer layers are executed.
            Loop 0 runs standard causal attention and stores KV caches.
            Loops 1+ run mixed attention with cross-loop processing.
        plt_window_size (`list` of `int`, *optional*, defaults to `[64, 0]`):
            Sliding window size `[left, right]` for the local attention in
            loop 1+. `[64, 0]` means a left-context window of 64 tokens with
            causal masking (right=0).
        plt_normalize_per_loop (`bool`, *optional*, defaults to `True`):
            When True, apply final_layernorm (shared weights) to hidden states
            at the end of each non-last loop before cross-loop processing.
        plt_emb_scale (`float`, *optional*, defaults to `None`):
            Scaling factor for the embedding in CLP: `a * E + b * shift(H)`.
            `None` means 1.0 (no scaling).
        plt_hidden_scale (`float`, *optional*, defaults to `None`):
            Scaling factor for the shifted hidden state in CLP:
            `a * E + b * shift(H)`. `None` means 1.0 (no scaling).
        plt_gate_use_hidden_states (`bool`, *optional*, defaults to `False`):
            Gate input mode. When `False`, the gate is computed as
            `sigmoid(einsum(Q, W_gate) + b_gate)` per head on the post-RoPE
            query tensor.  When `True`, gate uses
            `sigmoid(Linear(RMSNorm(hidden_states)))` (OLMo-style) instead.

    Example:
        ```python
        >>> from configuration_iquestpltcoder import IQuestPLTCoderConfig
        >>> from modeling_iquestpltcoder import IQuestPLTCoderModel

        >>> configuration = IQuestPLTCoderConfig()
        >>> model = IQuestPLTCoderModel(configuration)
        >>> configuration = model.config
        ```
    """

    model_type = "iquestpltcoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=75904,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=14,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=None,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        # PLT specific
        plt_num_loops=2,
        plt_window_size=None,
        plt_normalize_per_loop=True,
        plt_emb_scale=None,
        plt_hidden_scale=None,
        plt_gate_use_hidden_states=False,
        **kwargs,
    ):
        if eos_token_id is None:
            eos_token_id = [2, 75864, 75869]
        if plt_window_size is None:
            plt_window_size = [64, 0]

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # PLT specific
        self.plt_num_loops = plt_num_loops
        self.plt_window_size = plt_window_size
        self.plt_normalize_per_loop = plt_normalize_per_loop
        self.plt_emb_scale = plt_emb_scale
        self.plt_hidden_scale = plt_hidden_scale
        self.plt_gate_use_hidden_states = plt_gate_use_hidden_states

        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """Validate the `rope_scaling` configuration."""
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) < 1:
            raise ValueError(
                "`rope_scaling` must be a dictionary with a minimum of one field, "
                "`type` or `rope_type`."
            )

        rope_scaling_type = self.rope_scaling.get("type", None) or self.rope_scaling.get(
            "rope_type", None
        )
        if rope_scaling_type is None:
            raise ValueError("`rope_scaling` must have a `type` or `rope_type` field.")

        valid_rope_types = ["linear", "dynamic", "yarn", "longrope", "llama3"]
        if rope_scaling_type not in valid_rope_types:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of {valid_rope_types}, "
                f"got {rope_scaling_type}"
            )


__all__ = ["IQuestPLTCoderConfig"]
