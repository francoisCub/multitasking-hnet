from torch import nn, randn_like, cat, stack, permute
from math import ceil


class HnetChunked(nn.Module):
    def __init__(self, latent_size, output_size, batch, n_chunks):
        super().__init__()
        self.latent_size = latent_size
        self.n_chunks = n_chunks
        self.batch = batch
        self.output_size = output_size
        self.layer_embedding = nn.Linear(self.n_chunks, self.latent_size)
        self.layer_embedding.weight = nn.Parameter(
            randn_like(self.layer_embedding.weight))

        self.net = nn.Sequential(nn.Linear(2*latent_size, 2 * latent_size), nn.ReLU(),
                                 nn.Linear(2*latent_size, 2 * latent_size), nn.ReLU(),
                                 nn.Linear(2*latent_size, ceil(output_size/n_chunks)))
        
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.net[2].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.net[4].weight, nonlinearity='relu')

    def forward(self, z):
        # z could be task embedding
        if z.shape[1] != self.latent_size:
            raise ValueError()
        # z : b x latent_Size
        new_z = z.expand(self.n_chunks, z.shape[0], z.shape[1])
        # n x b x l
        new_z = permute(new_z, (1, 0, 2))
        # new_z : b x n x latent_Size
        embeddings = cat(
            [new_z, self.layer_embedding.weight.T.expand(new_z.shape)], dim=2)
        # new_z : b x n x 2latent_Size
        chunked_params = self.net(embeddings)
        # chunked_params : b x n x output_size/n
        params = stack([cat([theta for theta in chunks], dim=0)
                        for chunks in chunked_params])
        # chunked_params : b x output_size
        return params
