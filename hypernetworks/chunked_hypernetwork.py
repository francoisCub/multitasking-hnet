from turtle import forward
from torch import nn, randn_like, cat, stack


class HnetSparse(nn.Module):
    def __init__(self, latent_size, output_size, batch, n_chunks):
        super().__init__()
        if output_size % n_chunks != 0:
            raise ValueError()
        self.latent_size = latent_size
        self.n_chunks = n_chunks
        self.batch = batch
        self.layer_embedding = nn.Linear(self.n_chunks, self.latent_size)
        self.layer_embedding.weight = nn.Parameter(
            randn_like(self.layer_embedding.weight))

        self.net = nn.Sequential(   nn.Linear(latent_size, latent_size), nn.ReLu(),
                                    nn.Linear(latent_size, output_size//n_chunks))

    def forward(self, z):
        # z could be task embedding
        if self.batch:
            raise NotImplementedError()
        else:
            # z : 1 x latent_Size
            new_z = z.expand(self.n_chunk, z.shape[1])
            # new_z : n x latent_Size
            embeddings = cat([new_z, self.layer_embeddings], dim=1)
            # new_z : n x 2latent_Size
            chunked_params = self.net(embeddings)
            # chunked_params : n x output_size/n
            params = cat(chunked_params, dim=1)
            # chunked_params : 1 x output_size
