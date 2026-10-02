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


class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size,
            model_args.dim
        )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(model_args),
            persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id,
                model_args
            )

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )