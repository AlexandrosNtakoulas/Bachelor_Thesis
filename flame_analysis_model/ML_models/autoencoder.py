import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_input: int, num_latent: int, num_hidden: int) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_latent = num_latent
        self.num_hidden = num_hidden

        self.encode = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden//2),
            nn.ReLU(),
            nn.Linear(num_hidden//2, num_latent)
        )
    def forward(self, X):
        encoded = self.encode(X)
        return encoded

class Decoder(nn.Module):
    def __init__(self, num_input: int, num_latent: int, num_hidden: int) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_latent = num_latent
        self.num_hidden = num_hidden

        self.decode = nn.Sequential(
            nn.Linear(num_latent, num_hidden//2),
            nn.ReLU(),
            nn.Linear(num_hidden//2, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_input)
        )

    def forward(self, Y):
        decoded = self.decode(Y)
        return decoded

class AutoEncoder(nn.Module):
    def __init__(self, num_input: int, num_latent: int, num_hidden: int) -> None:
        super().__init__()
        self.num_input = num_input
        self.num_latent = num_latent
        self.num_hidden = num_hidden

        self.encoder = Encoder(
            num_input = self.num_input,
            num_latent = self.num_latent,
            num_hidden = self.num_hidden
        )
        self.decoder = Decoder(
            num_input = self.num_input,
            num_latent = self.num_latent,
            num_hidden = self.num_hidden
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def transform(self, X):
        '''Take X and encode to latent space'''
        encoded = self.encoder(X)
        return encoded

    def inverse_transform(self, Y):
        '''Take Y and decode to original space'''
        decoded = self.decoder(Y)
        return decoded
