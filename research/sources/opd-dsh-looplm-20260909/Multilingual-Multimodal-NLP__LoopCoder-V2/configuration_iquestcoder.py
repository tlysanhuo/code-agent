"""IQuestCoder model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class IQuestCoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IQuestCoderModel`]. It is used to instantiate
    an IQuestCoder model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 76800):
            Vocabulary size of the IQuestCoder model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`IQuestCoderModel`].
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 27648):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 40):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention (GQA).
            If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA).
            If `num_key_value_heads=1`, the model will use Multi Query Attention (MQA).
        head_dim (`int`, *optional*, defaults to 128):
            The dimension of each attention head. If not specified, defaults to `hidden_size // num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 500000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Supports various RoPE scaling
            types including "linear", "dynamic", "yarn", "longrope", etc.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        clip_qkv (`float`, *optional*):
            If set, clip the query, key, and value tensors to this value. Borrowed from OLMo for training stability.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention. Borrowed from Qwen2.
        sliding_window (`int`, *optional*):
            The sliding window size. Only effective when `use_sliding_window=True`.
        max_window_layers (`int`, *optional*, defaults to 0):
            The number of layers that don't use sliding window attention. Borrowed from Qwen2.

    Example:
        ```python
        >>> from configuration_iquestcoder import IQuestCoderConfig
        >>> from modeling_iquestcoder import IQuestCoderModel

        >>> # Initializing a IQuestCoder configuration
        >>> configuration = IQuestCoderConfig()

        >>> # Initializing a model from the configuration
        >>> model = IQuestCoderModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "iquestcoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Tensor / pipeline parallel plans for vLLM transformers backend.
    # Same shape as LlamaConfig — IQuestCoder is structurally Llama (RMSNorm + GQA + RoPE + SwiGLU).
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=76800,
        hidden_size=5120,
        intermediate_size=27648,
        num_hidden_layers=80,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=16384,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        # IQuestCoder specific (borrowed from OLMo)
        clip_qkv=None,
        # IQuestCoder specific (borrowed from Qwen2)
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=0,
        **kwargs,
    ):
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
        # IQuestCoder specific
        self.clip_qkv = clip_qkv
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # Validate rope_scaling configuration
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
                "`rope_scaling` must be a dictionary with a minimum of one field, `type` or `rope_type`."
            )
        
        rope_scaling_type = self.rope_scaling.get("type", None) or self.rope_scaling.get("rope_type", None)
        if rope_scaling_type is None:
            raise ValueError(
                "`rope_scaling` must have a `type` or `rope_type` field."
            )
        
        valid_rope_types = ["linear", "dynamic", "yarn", "longrope", "llama3"]
        if rope_scaling_type not in valid_rope_types:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of {valid_rope_types}, got {rope_scaling_type}"
            )


__all__ = ["IQuestCoderConfig"]

