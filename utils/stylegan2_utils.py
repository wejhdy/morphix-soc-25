import os
import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy

# --- CONFIG ---
NETWORK_PKL = os.path.join("stylegan2", "ffhq.pkl")
WPLUS_SHAPE = (18, 512)  # FFHQ StyleGAN2 W+ shape (no batch)


def load_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    w_plus_tensor = (
        torch.from_numpy(w_plus).unsqueeze(0).to(device)
    )  # [18, 512] -> [1, 18, 512]
    img = G.synthesis(w_plus_tensor, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return pil_img


def stylemix_wplus(wplus1, wplus2, layer_indices):
    """
    Mixes wplus1 and wplus2 at the specified layer indices.
    Args:
        wplus1: np.ndarray, shape (N, 512) or (18, 512)
        wplus2: np.ndarray, shape (N, 512) or (18, 512)
        layer_indices: list of int, layers to take from wplus2
    Returns:
        mixed: np.ndarray, same shape as input
    """
    assert wplus1.shape == wplus2.shape
    mixed = wplus1.copy()
    mixed[layer_indices] = wplus2[layer_indices]
    return mixed


def generate_image_custom(G, w_plus, truncation_psi=1.0, noise_mode="const"):
    """
    Generate a PIL image from a W+ latent with custom truncation psi and noise mode.
    Args:
        G: StyleGAN2 generator
        w_plus: np.ndarray, shape (N, 512) or (18, 512)
        truncation_psi: float
        noise_mode: str, one of 'const', 'random', 'none'
    Returns:
        PIL.Image
    """
    device = next(G.parameters()).device
    w_plus_tensor = torch.from_numpy(w_plus).unsqueeze(0).to(device)  # [N, 512] -> [1, N, 512]
    img = G.synthesis(w_plus_tensor, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    return pil_img


def stylemix_grid(G, wplus_list1, wplus_list2, layer_indices, truncation_psi=1.0, noise_mode="const"):
    """
    Generate a grid of style-mixed images for all pairs of wplus_list1 (rows) and wplus_list2 (cols).
    Args:
        G: StyleGAN2 generator
        wplus_list1: list of np.ndarray, each (N, 512)
        wplus_list2: list of np.ndarray, each (N, 512)
        layer_indices: list of int
        truncation_psi: float
        noise_mode: str
    Returns:
        grid_img: PIL.Image (grid of images)
        images: dict[(row_idx, col_idx)] = PIL.Image
    """
    n_rows = len(wplus_list1)
    n_cols = len(wplus_list2)
    images = {}
    for i, w1 in enumerate(wplus_list1):
        for j, w2 in enumerate(wplus_list2):
            mixed = stylemix_wplus(w1, w2, layer_indices)
            img = generate_image_custom(G, mixed, truncation_psi=truncation_psi, noise_mode=noise_mode)
            images[(i, j)] = img
    # Compose grid
    w, h = images[(0, 0)].size
    grid_img = PIL.Image.new('RGB', (w * n_cols, h * n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            grid_img.paste(images[(i, j)], (j * w, i * h))
    return grid_img, images
