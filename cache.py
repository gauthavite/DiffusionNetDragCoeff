import os
from multiprocessing import Pool
from functools import partial
from potpourri3d import read_mesh
from diffusion_utils import save_operators
from tqdm import tqdm
import torch
from config import Config

def prepopulate_cache_file(obj_file_path: str, cache_dir: str, n_eig: int) -> None:
    """
    Pre-populates the cache 1 mesh.

    Parameters:
    - obj_file_path (str): Path to the mesh file.
    - cache_dir (str): Path to the cache directory.
    - n_eig (int): Number of eigenvalues to use.
    """

    # Check if this mesh has been processed with this number of eigenvalues
    npz_file = obj_file_path.split("/")[-1].split(".")[0] + "_" + str(n_eig) + ".npz"
    save_path = os.path.join(cache_dir, npz_file)

    if not os.path.exists(save_path):
        verts, faces = read_mesh(obj_file_path)

        save_operators(torch.tensor(verts), torch.tensor(faces), n_eig, save_path)


def prepopulate_cache(
    data_basepath: str, cache_dir: str, n_eig: int, n_workers: int
) -> None:
    """
    Pre-populates the cache directory with all available meshes.

    Parameters:
    - data_basepath (str): Path to the base data directory.
    - cache_dir (str): Path to the cache directory.
    - n_eig (int): Number of eigenvectors to use for processing.
    - n_workers (int): Number of workers to use for data loading.
    """

    os.makedirs(cache_dir, exist_ok=True)

    # Gather all .obj files, excluding certain suffixes
    all_obj_files = [
        os.path.join(data_basepath, "meshes", f)
        for f in os.listdir(os.path.join(data_basepath, "meshes"))
        # if not (f.endswith("_flip.obj") or f.endswith("_aug.obj"))
    ]

    # Set up the worker function with partial to include constant arguments
    worker = partial(prepopulate_cache_file, cache_dir=cache_dir, n_eig=n_eig)

    # Use Pool with tqdm for a progress bar
    with Pool(processes=n_workers) as pool:
        # Wrap the pool.imap with tqdm for progress tracking
        for _ in tqdm(pool.imap(worker, all_obj_files), total=len(all_obj_files), desc="Caching files"):
            pass

if __name__ == '__main__':
# Cache data
    print("Caching data (might take a while)...")
    prepopulate_cache(
        Config.data_basepath,
        cache_dir=Config.cache_dir,
        n_eig=Config.num_eig,
        n_workers=Config.n_workers,
    )