from torch import nn, randn_like, cat, stack, permute


class HnetChunked(nn.Module):
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

        self.net = nn.Sequential(nn.Linear(2*latent_size, 2*latent_size), nn.ReLU(),
                                 nn.Linear(2*latent_size, 2 *
                                           latent_size), nn.ReLU(),
                                 nn.Linear(2*latent_size, output_size//n_chunks))

    def forward(self, z):
        # z could be task embedding
        if self.batch:
            if z.shape[1] != self.latent_size:
                raise ValueError()
            # z : b x latent_Size
            print(z.shape)
            new_z = z.expand(self.n_chunks, z.shape[0], z.shape[1])
            # n x b x l
            new_z = permute(new_z, (1, 0, 2))
            # new_z : b x n x latent_Size
            print(new_z.shape)
            embeddings = cat(
                [new_z, self.layer_embedding.weight.T.expand(new_z.shape)], dim=2)
            # new_z : b x n x 2latent_Size
            print(embeddings.shape)
            chunked_params = self.net(embeddings)
            # chunked_params : b x n x output_size/n
            print(chunked_params.shape)
            params = stack([cat([theta for theta in chunks], dim=0)
                           for chunks in chunked_params])
            print(params.shape)
            # chunked_params : b x output_size
            return params  # [1,0:self.output_size]
        else:
            if z.shape != (1, self.latent_size):
                raise ValueError()
            # z : 1 x latent_Size/2
            new_z = z.expand(self.n_chunks, z.shape[1])
            # new_z : n x latent_Size/2
            embeddings = cat([new_z, self.layer_embedding.weight.T], dim=1)
            # new_z : n x latent_Size
            chunked_params = self.net(embeddings)
            # chunked_params : n x output_size/n
            params = cat([theta for theta in chunked_params], dim=0)
            # chunked_params : 1 x output_size
            return params.unsqueeze(0)  # [1,0:self.output_size]
