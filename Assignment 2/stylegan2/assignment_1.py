import sys
import os
import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy


def init_random_vec_z():
    """Initialize a random vector z."""
    return np.random.RandomState().randn(1, 512)


def get_w_from_z(G, z, truncation_psi=0.8):
    """Get the W vector from a random vector z."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_tensor = torch.from_numpy(z).to(device)
    w = G.mapping(z_tensor, None, truncation_psi=truncation_psi)
    return w


def get_extended_w_plus(G, w):
    """Get the extended W+ vector."""
    w_plus = w.repeat(G.num_ws, 1, 1)
    return w_plus.cpu().numpy()


def generate_images(G, w_plus, outdir):
    """Generate images from the W+ vector."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_plus_tensor = torch.from_numpy(w_plus).to(device)

    img = G.synthesis(w_plus_tensor, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")


def main():
    network_pkl = "ffhq.pkl"
    outdir = "out"
    if len(sys.argv) >= 1:
        try:
            num_images = int(sys.argv[1])
        except ValueError:
            print("Invalid number of images specified. Using default value of 10.")
            num_images = 10

    # Check if the network_pkl file exists
    if not os.path.exists(network_pkl):
        print(f"Network file {network_pkl} does not exist.")
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    # Load the pre-trained StyleGAN2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)

    for i in range(num_images):
        # Generate a random vector Z
        z = init_random_vec_z()

        # Generate the W vector from Z
        w = get_w_from_z(G, z)

        # Generate and save the extended W+ vector from W
        w_plus = get_extended_w_plus(G, w)
        np.save(f"{outdir}/wplus_{i+1:02d}.npy", w_plus)

        # Generate and save the image
        image = generate_images(G, w_plus, outdir)
        image.save(f"{outdir}/image_{i+1:02d}.png")


if __name__ == "__main__":
    main()
