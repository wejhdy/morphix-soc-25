import os
import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy


def generate_and_save_random_image():
    """
    Generate a random image and its W+ vector using the pretrained StyleGAN2 model,
    then save them to the 'out' directory.
    """
    network_pkl = "ffhq.pkl"
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)
    # Generate a random vector Z
    z = np.random.RandomState().randn(1, G.z_dim)
    # Generate the W vector from Z
    z_tensor = torch.from_numpy(z).to(device)
    w = G.mapping(z_tensor, None, truncation_psi=0.8)
    # Generate the extended W+ vector from W
    w_plus = w.repeat(G.num_ws, 1, 1)
    w_plus_np = w_plus.cpu().numpy()
    np.save(f"{outdir}/wplus_random.npy", w_plus_np)
    # Generate and save the image
    w_plus_tensor = torch.from_numpy(w_plus_np).to(device)
    img = G.synthesis(w_plus_tensor, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = PIL.Image.fromarray(img[0].cpu().numpy(), "RGB")
    image.save(f"{outdir}/image_random.png")
    print(f"Generated image and wplus vector in '{outdir}' directory.")


if __name__ == "__main__":
    generate_and_save_random_image()
