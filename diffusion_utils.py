"""
Code taken from Nick Sharp at
https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/geometry.py
"""
import hashlib
import os
import numpy as np
import sklearn.neighbors
import scipy.sparse
import scipy.sparse.linalg as sla
import torch
import potpourri3d as pp3d

from tqdm.auto import tqdm

from potpourri3d import read_mesh

def read_off(filepath):
    """
    read a standard .off file

    Parameters
    -------------------------
    file : path to a '.off'-format file

    Output
    -------------------------
    vertices,faces : (n,3), (m,3) array of vertices coordinates
                    and indices for triangular faces
    """
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')
        n_verts, n_faces, _ = [int(x) for x in f.readline().strip().split(' ')]
        vertices = [[float(x) for x in f.readline().strip().split()] for _ in range(n_verts)]
        if n_faces > 0:
            faces = [[int(x) for x in f.readline().strip().split()][1:4] for _ in range(n_faces)]
            faces = np.asarray(faces)
        else:
            faces = None

    vertices = np.asarray(vertices)

    return torch.from_numpy(vertices), torch.from_numpy(faces).type(torch.LongTensor)

def normalize_positions(pos, faces=None, method='mean', scale_method='max_rad'):
    # center and unit-scale positions

    if method == 'mean':
        # center using the average point position
        pos = (pos - torch.mean(pos, dim=-2, keepdim=True))
    elif method == 'bbox': 
        # center via the middle of the axis-aligned bounding box
        bbox_min = torch.min(pos, dim=-2).values
        bbox_max = torch.max(pos, dim=-2).values
        center = (bbox_max + bbox_min) / 2.
        pos -= center.unsqueeze(-2)
    else:
        raise ValueError("unrecognized method")

    if scale_method == 'max_rad':
        scale = torch.max(norm(pos), dim=-1, keepdim=True).values.unsqueeze(-1)
        pos = pos / scale
    elif scale_method == 'area': 
        if faces is None:
            raise ValueError("must pass faces for area normalization")
        coords = pos[faces]
        vec_A = coords[:, 1, :] - coords[:, 0, :]
        vec_B = coords[:, 2, :] - coords[:, 0, :]
        face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
        total_area = torch.sum(face_areas)
        scale = (1. / torch.sqrt(total_area))
        pos = pos * scale
    else:
        raise ValueError("unrecognized scale method")
    return pos

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()


def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k+1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1]-1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def face_coords(verts, faces):
    coords = verts[faces]
    return coords

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = toNP(face_normals(torch.tensor(verts), torch.tensor(faces))) # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0: # point cloud

        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts_np[neigh_inds,:]
        neigh_points = neigh_points - verts_np[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)

    else: # mesh

        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)): raise ValueError("NaN normals :(")

    return normals


def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def build_tangent_frames(verts, faces, normals=None):

    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where((torch.abs(dot(vert_normals, basis_cand1))
                          < 0.9).unsqueeze(-1), basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
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

    edges_np = toNP(edges)
    edge_tangent_vectors_np = toNP(edge_tangent_vectors)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
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
            jV = edges_np[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.

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
        (data_vals, (row_inds, col_inds)), shape=(
            N, N)).tocsc()

    return mat


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat

def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.
    See get_operators() for a similar routine that wraps this one with a layer of caching.
    Torch in / torch out.
    Arguments:
      - vertices: (V,3) vertex positions
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
    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately, rather than as one complex sparse matrix.
    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix, like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    V = verts.shape[0]
    is_cloud = faces.numel() == 0

    eps = 1e-8

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)
    frames_np = toNP(frames)


    L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
    massvec_np = pp3d.vertex_areas(verts_np, faces_np)
    massvec_np += eps * np.mean(massvec_np)

    if(np.isnan(L.data).any()):
        raise RuntimeError("NaN Laplace matrix")
    if(np.isnan(massvec_np).any()):
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0])*eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise, but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma)

                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0., a_max=float('inf'))

                break
            except Exception as e:
                print(e)
                if(failcount > 3):
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (eps * 10**failcount)


    else: #k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0],0))


    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood

    edges = torch.tensor(np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat_np = build_grad(verts, edges, edge_vecs)


    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L, evals, evecs, gradX, gradY


def get_all_operators(verts_list, faces_list, k_eig, op_cache_dir=None, normals=None):
    N = len(verts_list)
            
    frames = [None] * N
    massvec = [None] * N
    L = [None] * N
    evals = [None] * N
    evecs = [None] * N
    gradX = [None] * N
    gradY = [None] * N

    inds = [i for i in range(N)]
    # process in random order
    # random.shuffle(inds)
   
    for num, i in enumerate(tqdm(inds, desc="get_all_operators")):
        #print("get_all_operators() processing {} / {} {:.3f}%".format(num, N, num / N * 100))
        if normals is None:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir)
        else:
            outputs = get_operators(verts_list[i], faces_list[i], k_eig, op_cache_dir, normals=normals[i])
        frames[i] = outputs[0]
        massvec[i] = outputs[1]
        L[i] = outputs[2]
        evals[i] = outputs[3]
        evecs[i] = outputs[4]
        gradX[i] = outputs[5]
        gradY[i] = outputs[6]
        
    return frames, massvec, L, evals, evecs, gradX, gradY


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def get_operators(verts, faces, k_eig=128, op_cache_dir=None, normals=None, overwrite_cache=False):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)
    is_cloud = faces.numel() == 0

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")

    # Check the cache directory
    # Note 1: Collisions here are exceptionally unlikely, so we could probably just use the hash...
    #         but for good measure we check values nonetheless.
    # Note 2: There is a small possibility for race conditions to lead to bucket gaps or duplicate
    #         entries in this cache. The good news is that that is totally fine, and at most slightly
    #         slows performance with rare extra cache misses.
    found = False
    if op_cache_dir is not None:
        ensure_dir_exists(op_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))
        # print("Building operators for input with hash: " + hash_key_str)

        # Search through buckets with matching hashes.  When the loop exits, this
        # is the bucket index of the file we should write to.
        i_cache_search = 0
        while True:

            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir,
                hash_key_str + "_" + str(i_cache_search) + ".npz")
            
            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache: 
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break
                
                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break
                
                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break


                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:,:k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(gradY).to(device=device, dtype=dtype)
                
                found = True
                
                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break
            
            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:

        # No matching entry found; recompute.
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k_eig, normals=normals)

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:

            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(search_path,
                     verts=verts_np.astype(dtype_np),
                     frames=toNP(frames).astype(dtype_np),
                     faces=faces_np,
                     k_eig=k_eig,
                     mass=toNP(mass).astype(dtype_np),
                     L_data = L_np.data.astype(dtype_np),
                     L_indices = L_np.indices,
                     L_indptr = L_np.indptr,
                     L_shape = L_np.shape,
                     evals=toNP(evals).astype(dtype_np),
                     evecs=toNP(evecs).astype(dtype_np),
                     gradX_data = gradX_np.data.astype(dtype_np),
                     gradX_indices = gradX_np.indices,
                     gradX_indptr = gradX_np.indptr,
                     gradX_shape = gradX_np.shape,
                     gradY_data = gradY_np.data.astype(dtype_np),
                     gradY_indices = gradY_np.indices,
                     gradY_indptr = gradY_np.indptr,
                     gradY_shape = gradY_np.shape,
                     )

    return frames, mass, L, evals, evecs, gradX, gradY

def save_operators(verts, faces, n_eig, save_path):
    """
    See documentation for compute_operators(). This essentailly just wraps a call to compute_operators, using a cache if possible.
    All arrays are always computed using double precision for stability, then truncated to single precision floats to store on disk, and finally returned as a tensor with dtype/device matching the `verts` input.
    """

    verts_np = toNP(verts)
    faces_np = toNP(faces)

    if(np.isnan(verts_np).any()):
        raise RuntimeError("tried to construct operators from NaN verts")
    
    frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, n_eig)
    dtype_np = np.float32

    L_np = sparse_torch_to_np(L).astype(dtype_np)
    gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
    gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

    np.savez(save_path,
                verts=verts_np.astype(dtype_np),
                frames=toNP(frames).astype(dtype_np),
                faces=faces_np,
                k_eig=n_eig,
                mass=toNP(mass).astype(dtype_np),
                L_data = L_np.data.astype(dtype_np),
                L_indices = L_np.indices,
                L_indptr = L_np.indptr,
                L_shape = L_np.shape,
                evals=toNP(evals).astype(dtype_np),
                evecs=toNP(evecs).astype(dtype_np),
                gradX_data = gradX_np.data.astype(dtype_np),
                gradX_indices = gradX_np.indices,
                gradX_indptr = gradX_np.indptr,
                gradX_shape = gradX_np.shape,
                gradY_data = gradY_np.data.astype(dtype_np),
                gradY_indices = gradY_np.indices,
                gradY_indptr = gradY_np.indptr,
                gradY_shape = gradY_np.shape,
                )
    print("Saved operators to " + save_path)
    return