import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ======= Model Definitions =======

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=784, label_dim=10, latent_dim=20):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim + label_dim, 400)
        self.fc_mu = torch.nn.Linear(400, latent_dim)
        self.fc_logvar = torch.nn.Linear(400, latent_dim)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=20, label_dim=10, output_dim=784):
        super().__init__()
        self.fc1 = torch.nn.Linear(latent_dim + label_dim, 400)
        self.fc_out = torch.nn.Linear(400, output_dim)

    def forward(self, z, y):
        z = torch.cat([z, y], dim=1)
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))

class ConditionalVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes=num_classes).float()

# ======= Load Model Weights =======
@st.cache_resource
def load_model():
    model = ConditionalVAE()
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ======= Streamlit App UI =======

st.title("🎨 MNIST Digit Generator")
digit = st.number_input("Enter a digit (0–9)", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    cols = st.columns(5)
    y = one_hot(torch.tensor([digit]))

    for i in range(5):
        z = torch.randn(1, 20)
        with torch.no_grad():
            img_tensor = model.decoder(z, y).view(28, 28).numpy()
        img_array = (img_tensor * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        cols[i].image(img, use_column_width=True, caption=f"Sample {i+1}")
