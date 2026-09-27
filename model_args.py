from dataclasses import dataclass, field
from loguru import logger
from torch import nn 

from torchfeather.model.moe import MoEArgs 

@dataclass 

class DeepSeekV3ModelArgs: 
    max_seq_len: int = 4096*4 
    vocab_size: int = 102400
    dim: int = 2048 
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    norm_eps: float = 1e-5

    moe_args: MoEArgs = field(default_factory=MoEArgs)

    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 128 

    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale:float = 1.0
asd