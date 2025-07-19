import os
import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy

# --- CONFIG ---
NETWORK_PKL = os.path.join('stylegan2', 'ffhq.pkl')
WPLUS_SHAPE = (18, 512)  # FFHQ StyleGAN2 W+ shape (no batch)

def load_generator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(NETWORK_PKL) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    return G, device

def random_wplus(G):
    z = np.random.randn(1, G.z_dim)
    z_tensor = torch.from_numpy(z).to(next(G.parameters()).device)
    w = G.mapping(z_tensor, None, truncation_psi=0.8)
    w_plus = w.repeat(G.num_ws, 1, 1).cpu().numpy()[0]  # [1, 18, 512] -> [18, 512]
    return w_plus

def generate_image(G, w_plus):
    device = next(G.parameters()).device
    w_plus_tensor = torch.from_numpy(w_plus).unsqueeze(0).to(device)  # [18, 512] -> [1, 18, 512]
    img = G.synthesis(w_plus_tensor, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return pil_img