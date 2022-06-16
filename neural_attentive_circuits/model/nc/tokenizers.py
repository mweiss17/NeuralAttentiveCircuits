from typing import Optional

import torch
import torch.nn as nn
import einops as eo
from torch.nn.functional import gumbel_softmax

from neural_compilers.model.external.convnext import ConvNeXt
from neural_compilers.model.nc.utils import PositionalGrid
from neural_compilers.utils import override


class InputTokenizer(nn.Module):
    @property
    def output_dim(self):
        raise NotImplementedError


class ConvNeXtImageTokenizer(InputTokenizer):
    PRESET_TO_DEPTHS = {
        "tiny": [3, 3, 9],
        "mini": [3, 3, 6],
        "micro": [3, 3, 3],
        "nano": [1, 1, 1],
        "pico": [1, 1, 0],
        "femto": [1, 0, 0],
        "atto": [0, 0, 0],
    }

    def __init__(
        self,
        input_dim: int = 3,
        repr_dim: int = 384,
        capacity_preset: str = "tiny",
        num_stages: int = 3,
        convnext_kwargs: Optional[dict] = None,
        positional_grid_kwargs: Optional[dict] = None,
    ):
        super(ConvNeXtImageTokenizer, self).__init__()
        depths = self.PRESET_TO_DEPTHS.get(capacity_preset, [3, 3, 27])[:num_stages]
        dims = [repr_dim // 4, repr_dim // 2, repr_dim][:num_stages]
        self.backbone = ConvNeXt(
            **override(
                convnext_kwargs or {},
                num_classes=None,
                gap=False,
                in_chans=input_dim,
                depths=depths,
                dims=dims,
            )
        )
        if positional_grid_kwargs is not None:
            self.positional_grid = PositionalGrid(**positional_grid_kwargs)
        else:
            self.positional_grid = None

    def forward(self, input: torch.Tensor):
        # input.shape = BCHW
        # features.shape = BCHW
        features = self.backbone(input)
        # Apply positional encodings (if required)
        if self.positional_grid is not None:
            positional_encodings = self.positional_grid(
                features.shape[0], tuple(features.shape[-2:])
            )
            features = torch.cat([features, positional_encodings], dim=1)
        # Tokenize
        features = eo.rearrange(features, "b c h w -> b (h w) c")
        return features

    @property
    def output_dim(self):
        return self.backbone.dims[-1] + (
            self.positional_grid.dim if self.positional_grid is not None else 0
        )


class OutputTokenizer(nn.Module):
    pass


class FirstSlotAsLogits(OutputTokenizer):
    def forward(self, states: torch.Tensor):
        if states.dim() == 3:
            # states.shape = BUC
            return states[:, 0, :]
        else:
            assert states.dim() == 2
            return states


class OutputLatentPooling(OutputTokenizer):
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        softmax_type: str = "gumbel",
        softmax_temperature: float = 1.0,
    ):
        super(OutputLatentPooling, self).__init__()
        # Attries
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.softmax_type = softmax_type
        self.softmax_temperature = softmax_temperature
        # Modules
        self.confidence_scorer = nn.Linear(state_dim, 1)
        self.to_logits = nn.Linear(state_dim, output_dim)

    def softmax(self, confidences: torch.Tensor):
        # confidences.shape = BU
        if self.softmax_type == "gumbel":
            confidences = gumbel_softmax(
                confidences,
                tau=self.softmax_temperature,
                hard=False,
                dim=-1,
            )
        elif self.softmax_type == "gumbel-hard":
            confidences = gumbel_softmax(
                confidences,
                tau=self.softmax_temperature,
                hard=True,
                dim=-1,
            )
        elif self.softmax_type == "vanilla":
            confidences = torch.softmax(confidences / self.softmax_temperature, dim=-1)
        else:
            raise NotImplementedError(f"Unknown softmax type: {self.softmax_type}")
        return confidences

    def forward(self, states: torch.Tensor):
        # states.shape = BUC
        # confidence.shape = BU
        confidences = self.confidence_scorer(states)[..., 0]
        # After the softmax, confidences is normalized along all the output latents.
        confidences = self.softmax(confidences)
        # Weight each state with its respective confidence
        # selected_states.shape = BC
        selected_states = torch.einsum("buc,bu->bc", states, confidences)
        # With the state selected, time to compute the logits
        logits = self.to_logits(selected_states)
        return logits
