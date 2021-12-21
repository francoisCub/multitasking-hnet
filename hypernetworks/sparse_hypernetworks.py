import math

import torch
from torch import nn
from torch_sparse import coalesce, spmm


class SparseLinear(nn.Module):
    def __init__(self, in_size, out_size, connectivity, distribution="uniform", sigma=torch.Tensor([1.0])):
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        indices_out = torch.LongTensor(list(range(out_size)) * connectivity)
        if distribution == "uniform":
            indices_in = (torch.randperm(out_size*connectivity) % in_size)
        elif distribution == "normal":
            indices_in = torch.round(torch.normal(torch.round(
                indices_out / (out_size / in_size)), sigma)) % in_size
        elif distribution == "mixed":
            if connectivity == 1:
                indices_in = torch.round(
                    indices_out / (out_size / in_size)) % in_size
            elif connectivity == 4:
                indices_in_minus_one = (torch.round(
                    indices_out / (out_size / in_size))-1) % in_size
                indices_in_aligned = (torch.round(
                    indices_out / (out_size / in_size))-1) % in_size
                indices_in_plus_one = (torch.round(
                    indices_out / (out_size / in_size))+1) % in_size
                indices_in = torch.round(torch.normal(torch.round(
                    indices_out / (out_size / in_size)), sigma)) % in_size
                indices_in[:out_size] = indices_in_minus_one[:out_size]
                indices_in[out_size:2*out_size] = indices_in_aligned[:out_size]
                indices_in[2*out_size:3 *
                           out_size] = indices_in_plus_one[:out_size]
            else:
                raise NotImplementedError(
                    "Mixed distribution not implemented for connectivity other than 1 or 4")
        else:
            raise ValueError("Distribution should be uniform or normal")
        indices = torch.stack([indices_out, indices_in])
        values = torch.randn(out_size * connectivity)
        indices, values = coalesce(indices.type(
            torch.long), values, out_size, in_size, op="add")
        self.register_buffer('indices', indices) #nn.Parameter(indices.type(torch.float))
        values = (torch.rand_like(values) - 0.5) * 2 * \
            math.sqrt(3 / (len(values)/out_size))
        self.values = nn.Parameter(values)

    def forward(self, x):
        return spmm(self.indices, self.values, self.out_size, self.in_size, x.t()).t()

    def extra_repr(self) -> str:
        return 'in_size={}, out_size={}'.format(
            self.in_size, self.out_size
        )


def sparse_helper(connectivity, distribution, sigma, first):
    if first:
        new_sigma = sigma.detach().clone() * 1e-4
        new_connectivity = 1 if distribution == "mixed" else connectivity
        first = False
    else:
        new_sigma = sigma
        new_connectivity = 4 if distribution == "mixed" else connectivity
    return new_connectivity, new_sigma, first


class HnetSparse(nn.Module):
    def __init__(self, latent_size, output_size, base=None, num_layers=None,
                 distribution="uniform", connectivity_type="constant", connectivity=3, sigma=torch.Tensor([1.0]), activation="none", step=1):
        super().__init__()
        self.latent_size = latent_size
        self.output_size = output_size
        if base is None:
            if num_layers is None:
                self.base = 2
                self.num_layers = math.ceil(
                    math.log(self.output_size / self.latent_size))
            else:  # num_layers
                self.num_layers = num_layers
                # real number
                self.base = (self.output_size /
                             self.latent_size) ** (1 / (self.num_layers))
        else:  # base
            if num_layers is not None:
                raise ValueError("Cannot set both base and num_layers")
            else:
                self.base = base
                self.num_layers = math.ceil(
                    math.log(self.output_size / self.latent_size))
        if distribution not in ["uniform", "normal", "mixed"]:
            raise ValueError(
                'distribution not in ["uniform", "normal", "mixed"]')
        self.distribution = distribution
        if connectivity_type not in ["linear-decrease", "exponential-decrease", "constant"]:
            raise ValueError(
                'connectivity_type not in ["linear-decrease", "exponential-decrease", "constant"]')
        self.connectivity_type = connectivity_type
        self.connectivity = connectivity
        self.sigma = sigma.detach().clone()

        layer_list = []
        current_size = self.output_size
        if connectivity_type in ["linear-decrease", "exponential-decrease"]:
            self.connectivity = 1
        first = True
        while(self.latent_size < current_size / self.base):
            new_connectivity, new_sigma, first = sparse_helper(
                self.connectivity, self.distribution, self.sigma, first)
            layer_list.append(SparseLinear(int(current_size / self.base),
                              current_size, new_connectivity, self.distribution, new_sigma))
            if activation == "leaky":
                layer_list.append(nn.LeakyReLU())
            elif activation == "relu":
                layer_list.append(nn.ReLU())
            elif activation == "prelu":
                layer_list.append(nn.PReLU(init=1.0))
            current_size = int(current_size / self.base)
            if connectivity_type == "linear-decrease":
                self.connectivity += step
            # self.sigma *= 2
        new_connectivity, new_sigma, first = sparse_helper(
            self.connectivity, self.distribution, self.sigma, first)
        layer_list.append(SparseLinear(
            self.latent_size, current_size, new_connectivity, self.distribution, new_sigma))

        self.net = nn.Sequential(*layer_list[::-1])

    def forward(self, x):
        return self.net(x)

    def extra_repr(self) -> str:
        return 'latent_size={}, output_size={}'.format(
            self.latent_size, self.output_size
        )


class Sparse(nn.Module):
    def __init__(self, in_size, out_size, indices_in, indices_out, init=None):
        super().__init__()
        if len(indices_in) != len(indices_out):
            raise ValueError(
                f"Should have same length of indices, got {len(indices_in)} and {len(indices_out)}")
        if isinstance(indices_in, list):
            indices_in = torch.LongTensor(indices_in)
        if isinstance(indices_out, list):
            indices_out = torch.LongTensor(indices_out)
        self.in_size = in_size
        self.out_size = out_size
        if init is None:
            v = torch.randn(len(indices_in)) * \
                (6 / math.sqrt(in_size+out_size))  # xavier-like
        else:
            v = torch.randn(len(indices_in)) * (6 / math.sqrt(init))
        self.weight = nn.Parameter(torch.sparse_coo_tensor(
            torch.stack([indices_out, indices_in]), v, (out_size, in_size)))
        self.weight.coalesce()

    def forward(self, x):
        return torch.sparse.mm(self.weight, x.T).T

    def extra_repr(self) -> str:
        return 'in_size={}, out_size={}'.format(
            self.in_size, self.out_size
        )


class Benes(nn.Module):
    def __init__(self, n, full=False, log=False):
        super().__init__()
        if n < 2 or n % 2 != 0:
            raise ValueError("n should be a power of 2")
        indices_in = [[0, 0, 1, 1]]
        indices_out = [[0, 1, 0, 1]]
        curr_n = 2
        while(curr_n < n):
            # increase width of networks
            for i in range(len(indices_in)):
                indices_in[i].extend([p+curr_n for p in indices_in[i]])
                indices_out[i].extend([p+curr_n for p in indices_out[i]])

            # add new layer
            sublist_low = list(range(curr_n)) * 2
            sublist_high = list(range(curr_n, curr_n*2)) * 2
            new_idx_in = sublist_low + sublist_high
            indices_in.append(new_idx_in)
            new_idx_out = list(range(curr_n*2)) * 2
            indices_out.append(new_idx_out)
            if full:
                indices_in.insert(0, new_idx_in.copy())
                indices_out.insert(0, new_idx_out.copy())

            curr_n *= 2

        # Build actual layer
        module_list = []
        for idx_in, idx_out in zip(indices_in, indices_out):
            if log:
                print("### ### ### ###")
                print(idx_in)
                print(idx_out)
            module_list.append(Sparse(n, n, idx_in, idx_out, init=4))

        self.net = nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)


class BenesOne(nn.Module):
    def __init__(self, latent_size, out_size, full=False, kernel_size=32):
        super().__init__()
        self.latent_size = latent_size
        self.out_size = out_size
        if kernel_size % 2 != 0 or kernel_size < 2:
            raise ValueError("kernel_size sould be an even number")

        next_latent = 2**(math.ceil(math.log(latent_size, 2)))

        before_out_size = 2**(math.ceil(math.log(out_size, 2)) - 1)

        self.adjuster = nn.Linear(self.latent_size, next_latent)

        # self.scaler = nn.Sequential(*[nn.ConvTranspose1d(1, 1, kernel_size=kernel_size, padding=kernel_size //
        #                             2-1, stride=2, bias=False) for _ in range(int(math.log(before_out_size/next_latent, 2)))])
        self.scaler = nn.Linear(next_latent, before_out_size)

        self.benes = Benes(before_out_size, full=full)

        # self.final = nn.Sequential(nn.ConvTranspose1d(
        #     1, 1, kernel_size=kernel_size, padding=kernel_size//2-1, stride=2, bias=False))
        self.final = nn.Linear(before_out_size, out_size)

    def forward(self, x):
        x = self.adjuster(x)
        x = self.scaler(x.unsqueeze(0))
        x = self.benes(x.squeeze(0))
        x = self.final(x.unsqueeze(0))
        return x.squeeze(0)[:, :self.out_size]

    def extra_repr(self) -> str:
        return 'latent_size={}, out_size={}'.format(
            self.latent_size, self.out_size
        )
