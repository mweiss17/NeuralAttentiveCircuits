from abc import abstractmethod, ABC
from typing import Optional, Mapping, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MixtureSameFamily, Categorical, kl, Bernoulli
import einops as eo

from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

from neural_compilers.utils import reduce_tensor, ScheduledHyperParameter


class WeightedSumLoss(nn.Module):
    def __init__(self, simplified_interface: bool = False):
        super(WeightedSumLoss, self).__init__()
        self.losses = nn.ModuleDict(dict())
        self.weights = nn.ModuleDict(dict())
        self.simplified_interface = simplified_interface

    def add_loss(
        self,
        loss: nn.Module,
        weight: Union[float, str, ScheduledHyperParameter],
        name: Optional[str] = None,
    ):
        assert name != "loss", "`loss` is a reserved name, please use something else."
        if name is None:
            name = f"loss_{len(self.losses)}"
            assert name not in self.losses
        self.losses[name] = loss
        if isinstance(weight, (float, int)):
            weight = ScheduledHyperParameter(float(weight))
        elif isinstance(weight, str):
            weight = ScheduledHyperParameter(1.0, schedule_spec=weight).update()
        else:
            assert isinstance(weight, ScheduledHyperParameter)
        self.weights[name] = weight
        return self

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss_terms = []
        unweighted_losses = {}
        for name, loss in self.losses.items():
            loss_value = loss(output, target)
            unweighted_losses[name] = loss_value
            loss_terms.append(loss_value * self.weights[name]())
        net_loss = torch.stack(loss_terms).sum()
        if self.simplified_interface:
            return net_loss
        else:
            assert "loss" not in unweighted_losses
            unweighted_losses["loss"] = net_loss
            return unweighted_losses


class SignatureHingeRepulsion(nn.Module):
    def __init__(self, model: nn.Module, hinge: float, reduction: str = "sum"):
        super(SignatureHingeRepulsion, self).__init__()
        self.model = model
        self.hinge = hinge
        self.reduction = reduction

    # noinspection PyUnusedLocal
    def forward(
        self,
        output: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def extract_signatures(latent_graph):
            input_signatures = latent_graph.input_latents.signatures
            mediator_signatures = latent_graph.mediator_latents.signatures
            output_signatures = latent_graph.output_latents.signatures
            return torch.cat(
                [input_signatures, mediator_signatures, output_signatures], dim=0
            )

        all_signatures = [
            extract_signatures(latent_graph)
            for latent_graph in self.model.latent_graphs
        ]
        all_kernels = [
            latent_graph.propagators[0].latent_attention.kernel
            for latent_graph in self.model.latent_graphs
        ]
        loss_terms = []
        tril_mask = None
        for signatures, kernel in zip(all_signatures, all_kernels):
            with kernel.return_distance():
                with kernel.do_not_sample_kernel():
                    # distance_matrix.shape = UU
                    distance_matrix = kernel(signatures, signatures)
            # Make the tril_mask if it's not already made
            if (tril_mask is None) or (tril_mask.shape != distance_matrix.shape):
                with torch.no_grad():
                    tril_mask = torch.tril(
                        torch.ones_like(distance_matrix, dtype=torch.bool), diagonal=-1
                    )
            # distances.shape = N
            distances = distance_matrix[tril_mask]
            # Select all distances that are below the threshold and whip them
            with torch.no_grad():
                too_close = distances.lt(self.hinge)
            distances = distances[too_close]
            loss_terms.append(-reduce_tensor(distances, mode=self.reduction, dim=0))
        loss = torch.stack(loss_terms).sum()
        return loss


class SignatureDistributionRegularization(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_components: int = 1,
        reduction: str = "mean",
        reduction_along_graphs: str = "sum",
    ):
        super(SignatureDistributionRegularization, self).__init__()
        self.model = model
        self.num_components = num_components
        self.reduction = reduction
        self.reduction_along_graphs = reduction_along_graphs
        self._compute_gaussian_components()

    @staticmethod
    def extract_signatures(latent_graph):
        input_signatures = latent_graph.input_latents.signatures
        mediator_signatures = latent_graph.mediator_latents.signatures
        output_signatures = latent_graph.output_latents.signatures
        return torch.cat(
            [input_signatures, mediator_signatures, output_signatures], dim=0
        )

    def _get_distances(self):
        ret_vals = []
        for latent_graph in self.model.latent_graphs:
            signatures = self.extract_signatures(latent_graph)
            kernel = latent_graph.propagators[0].latent_attention.kernel
            with kernel.return_distance():
                with kernel.do_not_sample_kernel():
                    distance_matrix = kernel(signatures, signatures)
            with torch.no_grad():
                tril_mask = torch.tril(
                    torch.ones_like(distance_matrix, dtype=torch.bool), diagonal=-1
                )
            distances = distance_matrix[tril_mask]
            ret_vals.append(
                dict(
                    signatures=signatures,
                    distances=distances,
                )
            )
        return ret_vals

    def _compute_gaussian_components(self):
        distances = [d["distances"] for d in self._get_distances()]
        mixture_weights = []
        mixture_means = []
        mixture_scales = []
        for _distances in distances:
            gmm = GaussianMixture(
                n_components=self.num_components, covariance_type="spherical"
            )
            gmm.fit(_distances.data.cpu().numpy()[:, None])
            mixture_means.append(torch.from_numpy(gmm.means_[:, 0]).float())
            mixture_scales.append(torch.from_numpy(gmm.covariances_).float().sqrt_())
            mixture_weights.append(torch.from_numpy(gmm.weights_).float())
        # component_{weights, means, sigmas}.shape = (num_layers, num_components)
        self.register_buffer("component_weights", torch.stack(mixture_weights))
        self.register_buffer("component_means", torch.stack(mixture_means))
        self.register_buffer("component_sigmas", torch.stack(mixture_scales))

    def forward(
        self,
        output: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distances = torch.stack([d["distances"] for d in self._get_distances()], dim=0)
        log_probs = self._compute_log_probs(distances)
        loss = reduce_tensor(
            reduce_tensor(-log_probs, mode=self.reduction, dim=1),
            mode=self.reduction_along_graphs,
            dim=0,
        )
        return loss

    def _compute_log_probs(self, distances: torch.Tensor) -> torch.Tensor:
        # distances.shape = (num_layers, num_pairs)
        component_distribution = Normal(
            loc=self.component_means, scale=self.component_sigmas
        )
        weight_distribution = Categorical(probs=self.component_weights)
        # component_log_prob.shape = (num_pairs, num_layers, num_components)
        component_log_prob = component_distribution.log_prob(
            eo.rearrange(distances, "layers pairs -> pairs layers ()")
        )
        # mixture_weight_log_prob.shape = (num_layers, num_components)
        mixture_weight_log_prob = torch.log_softmax(weight_distribution.logits, dim=-1)
        log_prob = eo.rearrange(
            torch.logsumexp(component_log_prob + mixture_weight_log_prob[None], dim=-1),
            "pairs layers -> layers pairs",
        )
        return log_prob


class LinkProbabilityRegularizer(nn.Module, ABC):
    def __init__(
        self,
        model: nn.Module,
        regularize_input_latents: bool = True,
        regularize_mediator_latents: bool = True,
        regularize_output_latents: bool = True,
        clip: Optional[float] = None,
        loss_type: str = "mse",
        multiplicative_matching_noise: Optional[float] = None,
    ):
        super(LinkProbabilityRegularizer, self).__init__()
        # Only supported for a single latent_graph for now
        assert (
            len(model.latent_graphs) == 1
        ), "Only support for a single latent graph for now."
        self.model = model
        self.regularize_input_latents = regularize_input_latents
        self.regularize_mediator_latents = regularize_mediator_latents
        self.regularize_output_latents = regularize_output_latents
        self.clip = clip
        self.loss_type = loss_type
        self.multiplicative_matching_noise = multiplicative_matching_noise
        # Get the graphon and register it as buffer
        self.register_buffer("sampled_graphon", self.get_sampled_graphon())

    @staticmethod
    def match_and_permute(
        input_adjacency: torch.Tensor,
        target_graphon: torch.Tensor,
        multiplicative_matching_noise: Optional[float] = None,
    ):
        # input_adjacency is assumed to be a matrix of probabilities, i.e.
        # containing values between 0 and 1.
        # input_adjacency.shape = UV
        # target_graphon.shape = UV
        # Construct cost matrix based on MSE between pairs of rows
        # cost_matrix.shape = UU
        with torch.no_grad():
            if multiplicative_matching_noise is not None:
                noise_tensor = (
                    torch.randn_like(input_adjacency)
                    .mul_(multiplicative_matching_noise)
                    .add_(1.0)
                )
                noised_input_adjacency = noise_tensor * input_adjacency
            else:
                noised_input_adjacency = input_adjacency
            cost_matrix = (
                (target_graphon[:, None, :] - noised_input_adjacency[None, :, :])
                .pow(2)
                .mean(-1)
            )
            target_idx, input_idx = linear_sum_assignment(
                cost_matrix=cost_matrix.detach().cpu().numpy(), maximize=False
            )
            # Permute the inputs
            input_idx = torch.from_numpy(input_idx).long().to(input_adjacency.device)
        input_adjacency = input_adjacency[input_idx]
        input_adjacency = input_adjacency[:, input_idx]
        return input_adjacency

    @property
    def num_regularized_latents(self) -> int:
        num = 0
        if self.regularize_input_latents:
            num += self.model.latent_graphs[0].input_latents.num_latents
        if self.regularize_mediator_latents:
            num += self.model.latent_graphs[0].mediator_latents.num_latents
        if self.regularize_output_latents:
            num += self.model.latent_graphs[0].output_latents.num_latents
        return num

    @property
    def cluster_size(self) -> int:
        assert self.num_regularized_latents % self.num_clusters == 0
        return self.num_regularized_latents // self.num_clusters

    @abstractmethod
    def get_sampled_graphon(self) -> torch.Tensor:
        ...

    @property
    def kernel_is_unique(self) -> bool:
        kernel_hashes = [
            (
                propagator.latent_attention.kernel.initial_bandwidth,
                propagator.latent_attention.kernel.truncation,
            )
            for propagator in self.model.latent_graphs[0].propagators
        ]
        kernel_learnable = any(
            [
                propagator.latent_attention.kernel.learnable_bandwidth
                for propagator in self.model.latent_graphs[0].propagators
            ]
        )
        return not kernel_learnable and all(
            [kernel_hash == kernel_hashes[0] for kernel_hash in kernel_hashes]
        )

    def _fetch_signatures(self) -> torch.Tensor:
        signatures = []
        if self.regularize_input_latents:
            signatures.append(self.model.latent_graphs[0].input_latents.signatures)
        if self.regularize_mediator_latents:
            signatures.append(self.model.latent_graphs[0].mediator_latents.signatures)
        if self.regularize_output_latents:
            signatures.append(self.model.latent_graphs[0].output_latents.signatures)
        return torch.cat(signatures, dim=0)

    def _get_link_probabilities(self) -> List[torch.Tensor]:
        # If we're learning the bandwidths, each propagator might learn a different one.
        # But if we're not, it will be wasted compute if we evaluate the matching multiple
        # times.
        kernels = []
        if self.kernel_is_unique:
            kernels.append(
                self.model.latent_graphs[0].propagators[0].latent_attention.kernel
            )
        else:
            for propagator in self.model.latent_graphs[0].propagators:
                kernels.append(propagator.latent_attention.kernel)
        # Get the sigs
        signatures = self._fetch_signatures()
        # Init buffer
        link_probas = []
        for kernel in kernels:
            # Compute the link probas
            with kernel.do_not_sample_kernel():
                link_proba = kernel(signatures, signatures)
            # link_proba.shape = UV
            assert link_proba.dim() == 2
            link_probas.append(link_proba)
        # Done
        return link_probas

    def _compute_matched_loss(self, link_probability: torch.Tensor):
        if self.loss_type == "mse":
            return self.compute_mse_loss(link_probability)
        elif self.loss_type == "kl":
            return self.compute_kl_loss(link_probability)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def compute_mse_loss(self, link_probability: torch.Tensor):
        if self.clip is None or self.clip == 0.0:
            # no-clip loss is just the simple MSE
            return F.mse_loss(link_probability, self.sampled_graphon)
        else:
            # We kill loss terms below the threshold specified by clip
            # unreduced_loss.shape = UV
            unreduced_loss = F.mse_loss(
                link_probability, self.sampled_graphon, reduction="none"
            )
            return torch.where(
                unreduced_loss < self.clip,
                torch.zeros_like(unreduced_loss).add_(self.clip),
                unreduced_loss,
            ).mean()

    def compute_kl_loss(self, link_probability: torch.Tensor, eps=1e-6):
        link_proba_dist = Bernoulli(link_probability.clamp(min=eps, max=1.0 - eps))
        sampled_graphon_dist = Bernoulli(self.sampled_graphon)
        if self.clip is None or self.clip == 0.0:
            # no-clip loss is just the simple MSE
            return kl.kl_divergence(sampled_graphon_dist, link_proba_dist).mean()
        else:
            # We kill loss terms below the threshold specified by clip
            # unreduced_loss.shape = UV
            unreduced_loss = kl.kl_divergence(sampled_graphon_dist, link_proba_dist)
            return torch.where(
                unreduced_loss < self.clip,
                torch.zeros_like(unreduced_loss).add_(self.clip),
                unreduced_loss,
            ).mean()

    def forward(
        self,
        output: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get the link probabilities; these are the adjacency matrices, one for each
        # unique kernel in the model.
        link_probabilities = self._get_link_probabilities()
        # Buffer for storing losses
        losses = []
        for link_probability in link_probabilities:
            # Permute link_probabilities based on the target graphon
            link_probability = self.match_and_permute(
                link_probability,
                self.sampled_graphon,
                multiplicative_matching_noise=self.multiplicative_matching_noise,
            )
            # Compute the MSE
            loss = self._compute_matched_loss(link_probability)
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss


class StochasticBlockModelRegularization(LinkProbabilityRegularizer):
    def __init__(
        self,
        model: nn.Module,
        num_clusters: int = 10,
        p_in: float = 0.9,
        p_out: float = 0.05,
        **super_kwargs,
    ):
        self.num_clusters = num_clusters
        self.p_in = p_in
        self.p_out = p_out
        super(StochasticBlockModelRegularization, self).__init__(model, **super_kwargs)

    def get_sampled_graphon(self) -> torch.Tensor:
        sampled_graphon = torch.zeros(
            self.num_regularized_latents, self.num_regularized_latents
        ).add_(self.p_out)
        num_clusters = self.num_clusters
        cluster_size = self.cluster_size
        for cluster_idx in range(num_clusters):
            sl = slice(cluster_idx * cluster_size, (cluster_idx + 1) * cluster_size)
            sampled_graphon[sl, sl] = self.p_in
        return sampled_graphon


class BarabassiAlbertRegularization(LinkProbabilityRegularizer):
    def __init__(
        self,
        model: nn.Module,
        new_edges_per_node: int = 2,
        **super_kwargs,
    ):
        self.new_edges_per_node = new_edges_per_node
        super(BarabassiAlbertRegularization, self).__init__(model, **super_kwargs)

    def get_sampled_graphon(self) -> torch.Tensor:
        # number of nodes / timesteps
        N = self.num_regularized_latents
        m = self.new_edges_per_node

        # dynamical exponent
        beta = 0.5
        k = np.arange(N + 1)
        degree_dist = m * (N / k) ** (beta)
        degree_dist = degree_dist[1:]
        magic = 1 / (m * 16)
        adj_dist = degree_dist * magic
        adj_dist = (adj_dist[:, np.newaxis] * (1 / np.sqrt(k)))[:, 1:]
        adj_dist = torch.tensor(adj_dist).float().clip(min=0.0, max=1.0)
        return adj_dist


class WattsStrogatzRegularization(LinkProbabilityRegularizer):
    def __init__(
        self,
        model: nn.Module,
        k: int = 32,
        p: float = 0.4,
        **super_kwargs,
    ):
        self.k = k
        self.p = p
        super(WattsStrogatzRegularization, self).__init__(model, **super_kwargs)

    def get_sampled_graphon(self) -> torch.Tensor:
        node_wise_random_connection_prob = (
            self.p * self.k / self.num_regularized_latents
        )
        ws_graphon = (
            np.ones(self.num_regularized_latents) * node_wise_random_connection_prob
        )
        kk = self.k // 2
        for i in range(self.num_regularized_latents):
            if i - kk < 0:
                ws_graphon[i, i - kk :] = 1 - self.p
                ws_graphon[i, : i + kk] = 1 - self.p
            if i + kk > self.num_regularized_latents:
                ws_graphon[i, : ((i + kk) % self.num_regularized_latents)] = 1 - self.p
            ws_graphon[i, i - kk : i + kk] = 1 - self.p
        ws_graphon = torch.tensor(ws_graphon).float().clip(min=0.0, max=1.0)
        return ws_graphon


class RingOfCliquesRegularization(LinkProbabilityRegularizer):
    def __init__(
        self,
        model: nn.Module,
        num_clusters: int = 10,
        p_in: float = 0.9,
        p_out: float = 0.05,
        **super_kwargs,
    ):
        self.num_clusters = num_clusters
        self.p_in = p_in
        self.p_out = p_out
        super(RingOfCliquesRegularization, self).__init__(model, **super_kwargs)

    def get_sampled_graphon(self) -> torch.Tensor:
        sampled_graphon = torch.zeros(
            self.num_regularized_latents, self.num_regularized_latents
        ).add_(self.p_out)
        num_clusters = self.num_clusters
        cluster_size = self.cluster_size

        for cluster_idx in range(num_clusters):
            sl = slice(cluster_idx * cluster_size, (cluster_idx + 1) * cluster_size)
            sampled_graphon[sl, sl] = self.p_in

            sl2 = slice((cluster_idx * cluster_size), (cluster_idx + 2) * cluster_size)
            sl3 = slice((cluster_idx * cluster_size), (cluster_idx * cluster_size) + 1)
            sampled_graphon[sl2, sl3] = 1.0
            sampled_graphon[sl3, sl2] = 1.0
        return sampled_graphon
