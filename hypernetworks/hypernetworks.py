
import math

import torch
from torch import nn, rand

from hypernetworks.modules import Selector
from hypernetworks.sparse_hypernetworks import BenesOne, HnetSparse


class HyperNetwork(nn.Module):
    def __init__(self, batch_target_model: nn.Module, input_type="learned", hnet="linear", latent_size=32,
                 encoder=None, mode="one_block", variation=False, batch=True, one_vector=False, num_tasks=None, aggregate=False,
                 distribution="normal", connectivity_type="linear-decrease", connectivity=3, sigma=torch.Tensor([2]), activation="prelu", step=1, base=2) -> None:
        super().__init__()
        # Init
        self.target_model = batch_target_model
        self.variation = variation
        self.one_vector = one_vector
        self.batch = batch
        self.aggregate = aggregate
        if self.batch:
            self.batch_size = batch_target_model.batch_size
        if (input_type == "input" or input_type == "input-type") and not self.batch:
            raise ValueError()

        # Core
        self.latent_size = latent_size
        self.input = input_type
        if self.input == "learned":
            self.z = nn.Parameter(rand(self.latent_size).unsqueeze(0).expand(
                self.batch_size, self.latent_size)) if self.batch else nn.Parameter(rand(1, self.latent_size))
        elif self.input == "input":
            if encoder is None:
                raise ValueError("Should have an encode for input hnet")
            self.encoder = encoder  # get_encoder('encoder-resnet')
        elif self.input == "task":
            if num_tasks is None:
                raise ValueError("For task hnet, give the number of task")
            self.num_tasks = num_tasks
            self.task_encoder = nn.Linear(self.num_tasks, self.latent_size)
            self.task_encoder.weight = nn.Parameter(
                torch.randn_like(self.task_encoder.weight))
        elif self.input == "input-task":
            self.num_tasks = num_tasks
            self.task_encoder = nn.Linear(self.num_tasks, self.num_tasks)
            self.task_encoder.weight = nn.Parameter(
                torch.randn_like(self.task_encoder.weight))
            self.mixer = nn.Bilinear(
                self.num_tasks, self.num_tasks, self.latent_size)
            self.encoder = encoder  # get_encoder('encoder-resnet')
        else:
            raise ValueError()
        self.layerNorm = nn.LayerNorm(
            self.latent_size, elementwise_affine=False)

        # Heads
        if hnet == "linear":
            self.core = nn.Linear(self.latent_size, self.latent_size)
            self.layer_heads = nn.ModuleList([nn.Linear(self.latent_size, target_size) for (
                _, _, target_size) in self.target_model.get_params_info()])
        elif hnet == "MLP":
            mid_size = 30
            self.core = nn.Sequential(nn.Linear(self.latent_size, mid_size), nn.PReLU(),
                                      nn.Linear(mid_size, mid_size),  nn.PReLU())
            self.layer_heads = nn.ModuleList([nn.Linear(mid_size, target_size) for (
                _, _, target_size) in self.target_model.get_params_info()])
        elif hnet == "sparse":
            idx = 0
            module_list = []
            for _, _, target_size in self.target_model.get_params_info():
                module_list.append(Selector(idx, target_size))
                idx += target_size
            self.layer_heads = nn.ModuleList(module_list)
            # idx = total_size
            self.core = HnetSparse(self.latent_size, idx, base=base, distribution=distribution,
                                   connectivity_type=connectivity_type, connectivity=connectivity, sigma=sigma, activation=activation, step=step)

        elif hnet == "benes":
            idx = 0
            module_list = []
            for _, _, target_size in self.target_model.get_params_info():
                module_list.append(Selector(idx, target_size))
                idx += target_size
            self.layer_heads = nn.ModuleList(module_list)
            self.core = BenesOne(self.latent_size, idx, full=True)

        else:
            raise ValueError("hnet should be in linear or MLP")

    def forward_hnet_batch(self, z):
        if z.shape[1] != self.latent_size:
            raise RuntimeError(
                f"Expected z to have size latent size {self.latent_size}, got {z.shape[1]}")
        z = self.core(z)
        batch_size = z.shape[0]
        batch_params = [[] for _ in range(batch_size)]
        params = []
        for (name, shape, _), head in zip(self.target_model.get_params_info(), self.layer_heads):
            params.append((name, head(z).view(batch_size, *shape)))
        for i in range(batch_size):
            batch_params[i] = [(name, param[i]) for (name, param) in params]
            batch_params[i] = dict(batch_params[i])

        return batch_params

    def forward_hnet(self, z):
        z = self.core(z)
        params = []
        for (name, shape, _), head in zip(self.target_model.get_params_info(), self.layer_heads):
            params.append((name, head(z).view(shape)))
        params = dict(params)

        return params

    def encode_input(self, x, z=None):
        if self.input == "learned":
            z = self.z
        elif self.input == "task":
            z = nn.functional.one_hot(z, num_classes=self.num_tasks)
            z = self.task_encoder(z.float())
            if self.batch:
                z = z.expand(x.shape[0], self.latent_size)
        elif self.input == "input":  # input
            if self.encoder is not None:
                z = self.encoder(x)
                if self.aggregate and self.training:
                    perm = torch.randperm(z.size(0))
                    z = z[perm[0]].unsqueeze(0).expand(
                        x.shape[0], self.latent_size)
        elif self.input == "input-task":
            if self.encoder is not None:
                task = nn.functional.one_hot(z, num_classes=self.num_tasks)
                task = self.task_encoder(task.float())
                task = task.expand(x.shape[0], self.num_tasks)
                input_z = self.encoder(x)
                z = self.mixer(input_z, task)

        if self.training and self.variation:
            z = z + torch.randn_like(z)

        z = self.layerNorm(z)

        return z

    def vector_to_params(params):
        raise NotImplementedError()

    def forward(self, x, z=None):
        # Compute z
        z = self.encode_input(x, z)
        if z.shape[-1] != self.latent_size:
            raise ValueError(f"Expected {self.latent_size}, got z of shape {z.shape}")

        # Compute params with hnet. (by batch or unique)
        params = self.forward_hnet_batch(
            z) if self.batch else self.forward_hnet(z)

        if self.one_vector:
            params = self.vector_to_params(params)

        # Return target model output
        return self.target_model(x, params)