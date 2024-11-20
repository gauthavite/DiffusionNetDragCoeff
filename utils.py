"""
Some code taken from Nick Sharp at
https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/geometry.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg as sla
import torch
import potpourri3d as pp3d
from sklearn.model_selection import KFold, train_test_split
from config import Config

def create_fold_splits(
    path: str,
    output_file: str,
    n_folds: int = 1,
    ratio: float = 0.8,
    seed: int = 42,
    n_eig: int = 32,
) -> None:
    """
    Splits data into train and test sets for n-fold cross-validation.

    Parameters:
    - path (str): Path to the .npz cached files.
    - output_file (str): Path to save the JSON output with the train/test split information.
    - train_ratio (float): Ratio of files to include in the train set if n=1.
    - n (int): Number of folds for cross-validation. If n=1, perform a single 80/20 split.
    - n_eig (int): Number of eigenvalues to use.
    """
    label_map = pd.read_csv(os.path.join(Config.data_basepath, "drag_coeffs.csv"))

    all_files = os.listdir(path)
    files = [f for f in all_files if '_'.join(f.split("_")[:-1]) in label_map.file.values and (int(f.split(".")[0].split("_")[-1]) == n_eig)]
    # files = [
    #     f.split("_")[0] for f in all_files if f.split("_")[0] in label_map.file.values
    # ]

    random.seed(seed)
    random.shuffle(files)

    split_data = {}
    
    if n_folds == 1:
        train_files, test_files = train_test_split(files, train_size=ratio, random_state=seed)
        split_data[f'fold_0'] = {'train': train_files, 'test': test_files}
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(kf.split(files)):
            train_files = [files[idx] for idx in train_index]
            test_files = [files[idx] for idx in test_index]
            split_data[f'fold_{i + 1}'] = {'train': train_files, 'test': test_files}
    
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=4)
    
    print(f"Split saved to {output_file}")


def cross(vec_A, vec_B):
    return np.cross(vec_A, vec_B, axis=-1)


def dot(vec_A, vec_B):
    return np.sum(vec_A * vec_B, axis=-1)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes normalized vectors along the last dimension of the input array.
    
    Parameters:
    - x: np.ndarray, the input array of shape (..., d), where d is the last dimension to normalize.
    - divide_eps: float, small value to avoid division by zero.
    - highdim: bool, indicates if the function should allow a high last dimension (>4).

    Returns:
    - np.ndarray, normalized array with the same shape as input.
    """
    
    if len(x.shape) == 1:
        raise ValueError(
            f"Called normalize() on single vector of dim {x.shape}. Are you sure?"
        )
    
    if not highdim and x.shape[-1] > 4:
        raise ValueError(
            f"Called normalize() with large last dimension {x.shape}. Are you sure?"
        )
    
    # Compute the norm along the last dimension and add epsilon
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + divide_eps
    
    return x / norms

def face_normals(verts, faces, normalized=True):
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


def mesh_vertex_normals(verts, faces):
    face_n = face_normals(verts, faces)

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=-1, keepdims=True
    )

    return vertex_normals


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def vertex_normals(verts, faces, n_neighbors_cloud=30):

    normals = mesh_vertex_normals(verts, faces)

    # if any are NaN, wiggle slightly and recompute
    bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
    if bad_normals_mask.any():
        bbox = np.amax(verts, axis=0) - np.amin(verts, axis=0)
        scale = np.linalg.norm(bbox) * 1e-4
        wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
        wiggle_verts = verts + bad_normals_mask * wiggle
        normals = mesh_vertex_normals(wiggle_verts, faces)

    # if still NaN assign random normals (probably means unreferenced verts in mesh)
    bad_normals_mask = np.isnan(normals).any(axis=1)
    if bad_normals_mask.any():
        normals[bad_normals_mask, :] = (
            np.random.RandomState(seed=777).rand(*verts.shape) - 0.5
        )[bad_normals_mask, :]
        normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    return normals


def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    dots = dots[..., np.newaxis]
    return vecs - unit_normals * dots


def build_tangent_frames(verts, faces, normals=None):
    """
    Builds an orthogonal coordinate frame for each vertex based on normals.

    Parameters:
    - verts: (V, 3) array of vertex positions.
    - faces: (F, 3) array of face indices for mesh connectivity.
    - normals: (V, 3) optional array of precomputed normals. If None, they are computed.

    Returns:
    - frames: (V, 3, 3) array of X, Y, and Z coordinate frames at each vertex.
    """

    V = verts.shape[0]
    dtype = verts.dtype

    if normals is None:
        vert_normals = vertex_normals(verts, faces)  # Compute vertex normals (V, 3)
    else:
        vert_normals = normals

    # Create candidate basis vectors
    basis_cand1 = np.tile(np.array([1, 0, 0], dtype=dtype), (V, 1))
    basis_cand2 = np.tile(np.array([0, 1, 0], dtype=dtype), (V, 1))

    # Select the appropriate basisX vector based on dot product with normal
    basisX = np.where(
        (np.abs(np.einsum('ij,ij->i', vert_normals, basis_cand1)) < 0.9).reshape(-1, 1),
        basis_cand1,
        basis_cand2
    )

    # Project basisX onto the tangent plane and normalize
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)

    # Calculate basisY as the cross product of normals and basisX
    basisY = np.cross(vert_normals, basisX)

    # Stack basisX, basisY, and vert_normals to form the frames
    frames = np.stack((basisX, basisY, vert_normals), axis=-2)

    if np.isnan(frames).any():
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient. All values pointwise.
    - edges: (2, E)
    """

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges.shape[1]):
        tail_ind = edges[0, iE]
        tip_ind = edges[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.0

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(N, N)
    ).tocsc()

    return mat


def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.
    Arguments:
      - verts: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use
    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient
    """

    eps = 1e-8

    # Compute frame
    frames = build_tangent_frames(verts, faces, normals=normals)

    # Compute Laplacian and mass vector
    L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
    massvec = pp3d.vertex_areas(verts, faces)
    massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # Compute the eigenbasis
    if k_eig > 0:
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        Mmat = scipy.sparse.diags(massvec)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # Computing eigenvalues and eigenvectors
                evals, evecs = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma)

                # Clip eigenvalues to remove small negative values due to numerical issues
                evals_np = np.clip(evals_np, a_min=0.0, a_max=float("inf"))
                break
            except Exception as e:
                print(e)
                if failcount > 3:
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)
    else:
        evals = np.zeros((0))
        evecs = np.zeros((verts.shape[0], 0))

    # Build gradient matrices
    edges = np.stack((inds_row, inds_col), axis=0)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat = build_grad(verts, edges, edge_vecs)

    # Split complex gradient matrix into two real sparse matrices
    gradX = np.real(grad_mat)
    gradY = np.imag(grad_mat)

    return frames, massvec, L, evals, evecs, gradX, gradY


def save_operators(
    verts, faces, k_eig=128, save_path=None
) -> None:
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    frames, mass, L, evals, evecs, gradX, gradY = compute_operators(
        verts, faces, k_eig
    )

    np.savez(
        save_path,
        verts=verts,
        frames=frames,
        faces=faces,
        k_eig=k_eig,
        mass=mass,
        L_data=L.data,
        L_indices=L.indices,
        L_indptr=L.indptr,
        L_shape=L.shape,
        evals=evals,
        evecs=evecs,
        gradX_data=gradX.data,
        gradX_indices=gradX.indices,
        gradX_indptr=gradX.indptr,
        gradX_shape=gradX.shape,
        gradY_data=gradY.data,
        gradY_indices=gradY.indices,
        gradY_indptr=gradY.indptr,
        gradY_shape=gradY.shape,
    )
    print("YES")

    return frames, mass, L, evals, evecs, gradX, gradY

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def get_geometry(file_path, device):
    
    npzfile = np.load(file_path, allow_pickle=True)
    verts = npzfile["verts"]
    faces = npzfile["faces"]
    k_eig = npzfile["k_eig"].item()

    def read_sp_mat(prefix):
        data = npzfile[prefix + "_data"]
        indices = npzfile[prefix + "_indices"]
        indptr = npzfile[prefix + "_indptr"]
        shape = npzfile[prefix + "_shape"]
        mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
        return mat

    frames = npzfile["frames"]
    mass = npzfile["mass"]
    L = read_sp_mat("L")
    evals = npzfile["evals"][:k_eig]
    evecs = npzfile["evecs"][:, :k_eig]
    gradX = read_sp_mat("gradX")
    gradY = read_sp_mat("gradY")

    frames = torch.from_numpy(frames).to(device=device)
    mass = torch.from_numpy(mass).to(device=device)
    L = sparse_np_to_torch(L).to(device=device)
    evals = torch.from_numpy(evals).to(device=device)
    evecs = torch.from_numpy(evecs).to(device=device)
    gradX = sparse_np_to_torch(gradX).to(device=device)
    gradY = sparse_np_to_torch(gradY).to(device=device)

    return verts, faces, frames, mass, L, evals, evecs, gradX, gradY