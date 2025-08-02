import os
import numpy as np

def get_default_directions():
    return {
        'smile': os.path.join('latent_directions', 'random', 'smile.npy'),
        'age': os.path.join('latent_directions', 'random', 'age.npy'),
        'gender': os.path.join('latent_directions', 'random', 'gender.npy'),
        'yaw': os.path.join('latent_directions', 'random', 'yaw.npy'),
        'pitch': os.path.join('latent_directions', 'random', 'pitch.npy'),
    }

def load_directions(directions_config=None):
    """
    Load latent directions from .npy files into a dictionary.
    Expects W+ directions (shape [18, 512]).
    Args:
        directions_config (dict): Optional. Mapping of direction names to file paths.
    Returns:
        dict: Mapping of direction names to numpy arrays.
    """
    if directions_config is None:
        directions_config = get_default_directions()
    directions = {}
    for name, path in directions_config.items():
        arr = np.load(path)
        if arr.shape == (1, 18, 512):
            arr = arr.squeeze(0)
        elif arr.shape == (1, 1, 512):
            arr = np.tile(arr.squeeze(0), (18, 1))
        elif arr.shape == (512,):
            arr = np.tile(arr, (18, 1))
        directions[name] = arr
    return directions 