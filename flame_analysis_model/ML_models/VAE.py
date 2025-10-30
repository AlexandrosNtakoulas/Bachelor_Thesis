import torch
import torch.nn as nn
from .autoencoder import AutoEncoder

class VAE(nn.Module):
    def __init__(self, num_input: int, num_latent: int, num_hidden: int):
        super().__init__()

        # Encoder
        self.in_2hid = nn.Linear(num_input, num_hidden)
        self.hid_2mu  = nn.Linear(num_hidden, num_latent)
        self.hid_2sig = nn.Linear(num_hidden, num_latent)
        # Decoder
        self.z_2hid = nn.Linear(num_latent, num_hidden)
        self.hid_2in = nn.Linear(num_hidden, num_input)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.in_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sig(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2in(h))
    def forward(self, x):

        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_parametarized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_parametarized)
        return z_parametarized, x_reconstructed, mu, sigma

