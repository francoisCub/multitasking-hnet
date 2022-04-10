
import math

import torch
from torch import nn, rand

from hypernetworks.modules import Selector
from hypernetworks.sparse_hypernetworks import BenesOne, HnetSparse
from hypernetworks.chunked_hypernetwork import HnetChunked


class HyperNetwork(nn.Module):
    def __init__(self, batch_target_model: nn.Module, input_type="learned", hnet="linear", latent_size=32,
                 encoder=None, mode="one_block", variation=False, batch=True, one_vector=False, num_tasks=None, aggregate=False,
                 distribution="normal", connectivity_type="linear-decrease", connectivity=3, sigma=torch.Tensor([2]), activation="prelu", step=1, base=2, nbr_chunks=8, bias_sparse=False, normalize=True, post_processing=None) -> None:
        super().__init__()
        # Init
        self.target_model = batch_target_model
        self.variation = variation
        self.one_vector = one_vector
        self.batch = batch
        self.aggregate = aggregate
        if self.batch:
            self.batch_size = batch_target_model.batch_size
        if (input_type == "input" or input_type == "input-task") and not self.batch:
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
                self.latent_size, self.num_tasks, self.latent_size)
            self.encoder = encoder  # get_encoder('encoder-resnet')
        else:
            raise ValueError()
        if normalize:
            self.layerNorm = nn.LayerNorm(
            self.latent_size, elementwise_affine=False)
        else:
            self.layerNorm = nn.Identity()
        
        # Layers heads
        idx = 0
        module_list = []
        for _, shape, target_size in self.target_model.get_params_info():
            if post_processing is None:
                module_list.append(Selector(idx, target_size, shape=shape))
            else:
                # Post processing (used for target network sparsity for example)
                module_list.append(nn.Sequential(Selector(idx, target_size, shape=shape),
                                    post_processing(idx, target_size, shape=shape)))
            idx += target_size
        total_target_size = idx
        self.layer_heads = nn.ModuleList(module_list)

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
            # idx = total_size
            self.core = HnetSparse(self.latent_size, total_target_size, base=base, distribution=distribution,
                                   connectivity_type=connectivity_type, connectivity=connectivity, sigma=sigma, activation=activation, step=step, bias=bias_sparse)

        elif hnet == "benes":
            self.core = BenesOne(self.latent_size, total_target_size, full=True)
        
        elif hnet == "chunked":
            # raise NotImplementedError()
            self.core = HnetChunked(self.latent_size, total_target_size, batch=batch, n_chunks=nbr_chunks)
            
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
    
    def forward_average_z_model(self, x):
        with torch.no_grad():
            if not hasattr(self, "num_tasks") or not (self.input == "task"):
                raise ValueError("Average only for task-dependent models")
            mean_z = torch.zeros((1, self.latent_size), device=x.device)
            for task in range(self.num_tasks):
                z_t = self.encode_input(x, torch.tensor(task, dtype=torch.long, device=x.device))
                mean_z += z_t
            mean_z /= self.num_tasks
            params = self.forward_hnet_batch(
                mean_z) if self.batch else self.forward_hnet(mean_z)

            # Return target model output
            return self.target_model(x, params)
