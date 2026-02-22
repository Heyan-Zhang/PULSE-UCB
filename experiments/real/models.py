import torch.nn as nn


class InferenceModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Autoencoder_BN(nn.Module):
    def __init__(self, raw_dim, emb_dim):
        super().__init__()
        self.raw_dim = raw_dim
        self.emb_dim = emb_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.raw_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
        )
        self.decoder = nn.Linear(self.emb_dim, self.raw_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(x.shape[0], -1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size, self.raw_dim)
        return out

    def encoding_result(self, x):
        x = x.view(x.shape[0], -1)
        return self.encoder(x)
