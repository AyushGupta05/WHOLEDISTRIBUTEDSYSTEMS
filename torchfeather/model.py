import math
from typing import Optional

import torch
from torch import nn

from torchfeather.model.attention import (
    ScaledDotProductAttentionWrapper,
)
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim  # 2048
        self.n_heads = model_args.n_heads  # 16
        self.q_lora_rank = model_args.q_lora_rank  # 0
        self.kv_lora_rank = model_args.kv_lora_rank  # 512
        self.qk_nope_head_dim = model_args.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = model_args.qk_rope_head_dim  # 64

        self.qk_head_dim = (
            model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        )  # 128 + 64 = 192

        self.v_head_dim = model_args.v_head_dim  # 128

        
class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size,
            model_args.dim,
        )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(model_args),
            persistent=False,
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id,
                model_args,
            )

        self.norm = nn.RMSNorm(model_args.dim)

        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )

    def init_weights(self,init_std: Optional[float] = None,buffer_device: Optional[torch.device] = None,  ):
        buffer_device = buffer_device or self.freqs_cis.device

        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(
                    init_std=init_std,
                    buffer_device=buffer_device,
                )  # ty: ignore[call-non-callable]

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )