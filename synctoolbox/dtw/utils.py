import numpy as np
from libfmp.c3 import compute_strict_alignment_path_mask

from synctoolbox.dtw.core import compute_warping_path
from synctoolbox.dtw.cost import *
from synctoolbox.feature.utils import shift_chroma_vectors


def compute_optimal_chroma_shift(f_chroma1: np.ndarray,
                                 f_chroma2: np.ndarray,
                                 chroma_transpositions: np.ndarray = np.arange(0, 12),
                                 step_sizes: np.ndarray = np.array([[1, 0], [0, 1], [1, 1]], int),
                                 step_weights: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64)) -> int:
    """Computes the optimal chroma shift which minimizes the DTW cost.

    Parameters
    ----------
    f_chroma1 : np.ndarray [shape=(d_chroma, N_chroma)]
        First chroma vector

    f_chroma2 : np.ndarray [shape=(d_chroma, N_chroma)]
        Second chroma vector

    step_sizes : np.ndarray
        DTW step sizes (default: np.array([[1, 0], [0, 1], [1, 1]]))

    step_weights : np.ndarray
        DTW step weights (default: np.array([1.0, 1.0, 1.0]))

    chroma_transpositions : np.ndarray
        Array of chroma shifts (default: np.arange(0, 11))

    Returns
    -------
    opt_chroma_shift : int
        Optimal chroma shift which minimizes the DTW cost.
    """
    opt_chroma_shift = 0
    dtw_cost = np.inf
    for chroma_shift in chroma_transpositions:
        cost_matrix_tmp = cosine_distance(f_chroma1, shift_chroma_vectors(f_chroma2, chroma_shift))
        D, _, _ = compute_warping_path(cost_matrix_tmp, step_sizes=step_sizes, step_weights=step_weights)
        if D[-1, -1] < dtw_cost:
            dtw_cost = D[-1, -1]
            opt_chroma_shift = chroma_shift

    return opt_chroma_shift


def compute_warping_paths_from_cost_matrices(cost_matrices: list,
                                             step_sizes: np.array = np.array([[1, 0], [0, 1], [1, 1]], int),
                                             step_weights: np.array = np.array([1.0, 1.0, 1.0], np.float64),
                                             implementation: str = 'synctoolbox') -> list:
    """Computes a path via DTW on each matrix in cost_matrices

    Parameters
    ----------
    cost_matrices : list
        List of cost matrices

    step_sizes : np.ndarray
        DTW step sizes (default: np.array([[1, 0], [0, 1], [1, 1]]))

    step_weights : np.ndarray
        DTW step weights (default: np.array([1.0, 1.0, 1.0]))

    implementation : str
        Choose among 'synctoolbox' and 'librosa' (default: 'synctoolbox')

    Returns
    -------
    wp_list : list
        List of warping paths
    """
    return [compute_warping_path(C=C,
                                 step_sizes=step_sizes,
                                 step_weights=step_weights,
                                 implementation=implementation)[2] for C in cost_matrices]


def compute_cost_matrices_between_anchors(f_chroma1: np.ndarray,
                                          f_chroma2: np.ndarray,
                                          anchors: np.ndarray,
                                          f_DLNCO1: np.ndarray = None,
                                          f_DLNCO2: np.ndarray = None) -> list:
    """Computes cost matrices for the given features between subsequent
    pairs of anchors points.

    Parameters
    ----------
    f_chroma1 : np.ndarray [shape=(12, N)]
        Chroma feature matrix of the first sequence

    f_chroma2 : np.ndarray [shape=(12, M)]
        Chroma feature matrix of the second sequence

    anchors : np.ndarray [shape=(2, R)]
        Anchor sequence

    f_DLNCO1 : np.ndarray [shape=(12, N)]
        DLNCO feature matrix of the first sequence

    f_DLNCO2 : np.ndarray [shape=(12, M)]
        DLNCO feature matrix of the second sequence

    Returns
    -------
    cost_matrices: list
        List containing cost matrices
    """
    high_res = False
    if f_DLNCO1 is not None and f_DLNCO2 is not None:
        high_res = True

    cost_matrices = list()
    for k in range(anchors.shape[1] - 1):
        a1 = np.array(anchors[:, k].astype(int), copy=True)
        a2 = np.array(anchors[:, k + 1].astype(int), copy=True)

        if high_res:
            if f_chroma1.shape != f_DLNCO1.shape or f_chroma2.shape != f_DLNCO2.shape:
                raise ValueError('Chroma features and DLNCO features must be of the same shape.')
            cost_matrices.append(compute_high_res_cost_matrix(f_chroma1[:, a1[0]: a2[0] + 1],
                                                              f_chroma2[:, a1[1]: a2[1] + 1],
                                                              f_DLNCO1[:, a1[0]: a2[0] + 1],
                                                              f_DLNCO2[:, a1[1]: a2[1] + 1]))
        else:
            cost_matrices.append(cosine_distance(f_chroma1[:, a1[0]: a2[0] + 1],
                                                 f_chroma2[:, a1[1]: a2[1] + 1]))
    return cost_matrices


def build_path_from_warping_paths(warping_paths: list,
                                  anchors: np.ndarray = None) -> np.ndarray:
    """The function builds a path from a given list of warping paths
    and the anchors used to obtain these paths. The indices of the original
    warping paths are adapted such that they cross the anchors.

    Parameters
    ----------
    warping_paths : list
        List of warping paths

    anchors : np.ndarray [shape=(2, N)]
        Anchor sequence

    Returns
    -------
    path : np.ndarray [shape=(2, M)]
        Merged path
    """

    if anchors is None:
        # When no anchor points are given, we can construct them from the
        # subpaths in the wp_list

        # To do this, we assume that the first path's element is the starting
        # anchor
        anchors = warping_paths[0][:, 0]

        # Retrieve the last element of each path
        anchors_tmp = np.zeros(len(warping_paths), np.float32)
        for idx, x in enumerate(warping_paths):
            anchors_tmp[idx] = x[:, -1]

        # Correct indices, such that the indices of the anchors are given on a
        # common path. Each anchor a_l = [Nnew_[l+1];Mnew_[l+1]]
        #    Nnew_[l+1] = N_l + N_[l+1] -1
        #    Mnew_[l+1] = M_l + M_[l+1] -1

        anchors_tmp = np.cumsum(anchors_tmp, axis=1)
        anchors_tmp[:, 1:] = anchors_tmp[:, 1:] - [np.arange(1, anchors_tmp.shape[1]),
                                                   np.arange(1, anchors_tmp.shape[1])]

        anchors = np.concatenate([anchors, anchors_tmp], axis=1)

    L = len(warping_paths) + 1
    path = None
    wp = None

    for anchor_idx in range(1, L):
        anchor1 = anchors[:, anchor_idx - 1]
        anchor2 = anchors[:, anchor_idx]

        wp = np.array(warping_paths[anchor_idx - 1], copy=True)

        # correct indices in warpingPath
        wp += np.repeat(anchor1.reshape(-1, 1), wp.shape[1], axis=1).astype(wp.dtype)

        # consistency checks
        assert np.array_equal(wp[:, 0], anchor1), 'First entry of warping path does not coincide with anchor point'
        assert np.array_equal(wp[:, -1], anchor2), 'Last entry of warping path does not coincide with anchor point'

        if path is None:
            path = np.array(wp[:, :-1], copy=True)
        else:
            path = np.concatenate([path, wp[:, :-1]], axis=1)

    # append last index of warping path
    path = np.concatenate([path, wp[:, -1].reshape(-1, 1)], axis=1)

    return path


def find_anchor_indices_in_warping_path(warping_path: np.ndarray,
                                        anchors: np.ndarray):
    """Compute the indices in the warping path that corresponds
    to the elements in 'anchors'

    Parameters
    ----------
    warping_path : np.ndarray [shape=(2, N)]
        Warping path

    anchors : np.ndarray [shape=(2, M)]
        Anchor sequence

    Returns
    -------
    indices : np.ndarray [shape=(2, M)]
        Anchor indices in the ``warping_path``
    """
    indices = np.zeros(anchors.shape[1])

    for k in range(anchors.shape[1]):
        a = anchors[:, k]
        indices[k] = np.where((a[0] == warping_path[0, :]) & (a[1] == warping_path[1, :]))[0]

    return indices


def make_path_strictly_monotonic(P: np.ndarray) -> np.ndarray:
    """Compute strict alignment path from a warping path

    Wrapper around "compute_strict_alignment_path_mask" from libfmp.

    Parameters
    ----------
    P: np.ndarray [shape=(2, N)]
        Warping path

    Returns
    -------
    P_mod: np.ndarray [shape=(2, M)]
        Strict alignment path, M <= N
    """
    P_mod = compute_strict_alignment_path_mask(P.T)

    return P_mod.T
