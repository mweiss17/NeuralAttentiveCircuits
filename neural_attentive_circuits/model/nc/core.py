from typing import List, Mapping, Optional, Union

import torch
import torch.nn as nn

import neural_compilers.model.nc.tokenizers as tkn
import neural_compilers.model.nc.block as blk

from neural_compilers.model.nc.block import LatentGraph
from neural_compilers.model.nc.tokenizers import InputTokenizer, OutputTokenizer
from neural_compilers.utils import keep_valid_backups


class _NeuralCompiler(nn.Module):
    def __init__(
        self,
        input_tokenizer: InputTokenizer,
        latent_graphs: List[LatentGraph],
        output_tokenizer: OutputTokenizer,
    ):
        super(_NeuralCompiler, self).__init__()
        # We'll need to think a bit harder about how to stack multiple latent graphs.
        # Until then, we support only one.
        assert len(latent_graphs) == 1, "Only a single latent graph supported for now."
        self.latent_graphs = nn.ModuleList(latent_graphs)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # --------------------
        # First step is to tokenize the raw inputs (which can be BCHW
        # images or anything, doesn't matter).
        # We do provide a way to provide pre-tokenized inputs, which might be useful
        # if we want to tokenize elsewhere.
        raw_inputs = inputs.get("raw_inputs")
        if raw_inputs is not None:
            tokenized_inputs = self.input_tokenizer(raw_inputs)
        else:
            assert "tokenized_inputs" in inputs
            tokenized_inputs = inputs["tokenized_inputs"]
        # tokenized_inputs.shape = BUC
        assert tokenized_inputs.dim() == 3
        # --------------------
        # Next step is to pass it though the latent graph
        # TODO: Figure out how to stack multiple such layers
        latent_graph = self.latent_graphs[0]
        latent_graph_outputs = latent_graph(inputs=tokenized_inputs)
        # --------------------
        # Final step is to pass the output states through the output tokenizer
        outputs = self.output_tokenizer(latent_graph_outputs["output_states"])
        # Done
        return dict(
            raw_inputs=raw_inputs,
            tokenized_inputs=tokenized_inputs,
            outputs=outputs,
            **latent_graph_outputs,
        )


class NeuralCompiler(_NeuralCompiler):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        state_dim: int,
        input_tokenizer_type: str = "ConvNeXtImageTokenizer",
        input_tokenizer_kwargs: Optional[dict] = None,
        latent_graph_type: str = "LatentGraph",
        latent_graph_kwargs: Optional[dict] = None,
        output_tokenizer_type: str = "FirstSlotAsLogits",
        output_tokenizer_kwargs: Optional[dict] = None,
        simplified_interface: bool = False,
    ):
        # --------------------
        # Init the input tokenizer
        #   cls
        input_tokenizer_cls = getattr(tkn, input_tokenizer_type)
        #   kwargs
        input_tokenizer_kwargs = keep_valid_backups(
            input_tokenizer_cls,
            input_tokenizer_kwargs,
            input_dim=input_dim,
            repr_dim=state_dim,
        )
        #   module
        input_tokenizer = input_tokenizer_cls(**input_tokenizer_kwargs)
        # --------------------
        # Init the latent graph
        #   cls
        latent_graph_cls = getattr(blk, latent_graph_type)
        #   kwargs
        latent_graph_input_dim = input_tokenizer.output_dim
        latent_graph_kwargs = keep_valid_backups(
            latent_graph_cls,
            latent_graph_kwargs,
            state_dim=state_dim,
            input_dim=latent_graph_input_dim,
            output_dim=output_dim,
        )
        #   module
        latent_graphs = [latent_graph_cls(**latent_graph_kwargs)]
        # --------------------
        # Init the output tokenizer
        #   cls
        output_tokenizer_cls = getattr(tkn, output_tokenizer_type)
        #   kwargs
        output_tokenizer_kwargs = keep_valid_backups(
            output_tokenizer_cls,
            output_tokenizer_kwargs,
            state_dim=state_dim,
            output_dim=output_dim,
        )
        #   module
        output_tokenizer = output_tokenizer_cls(**output_tokenizer_kwargs)
        # --------------------
        # Init the super
        super(NeuralCompiler, self).__init__(
            input_tokenizer=input_tokenizer,
            latent_graphs=latent_graphs,
            output_tokenizer=output_tokenizer,
        )
        # --------------------
        # Extra kwargs for this class
        self.simplified_interface = simplified_interface

    def forward(
        self, *inputs: List[Union[Mapping[str, torch.Tensor], torch.Tensor]]
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        if self.simplified_interface:
            # This is the tensor-in-tensor-out (tito?) interface
            assert len(inputs) == 1
            assert torch.is_tensor(inputs[0])
            super_inputs = dict(raw_inputs=inputs[0])
        else:
            assert len(inputs) == 1
            assert isinstance(inputs[0], dict)
            super_inputs = inputs[0]
        # noinspection PyTypeChecker
        super_outputs = super(NeuralCompiler, self).forward(super_inputs)
        if self.simplified_interface:
            # tito interface means we'll have to return the right output
            # tensor and nothing else
            outputs = super_outputs["outputs"]
        else:
            outputs = super_outputs
        return outputs
