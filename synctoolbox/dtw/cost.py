from numba import jit
import numpy as np


#@jit(nopython=True)
def cosine_distance(f1, f2, cos_meas_max=2.0, cos_meas_min=1.0):
    """For all pairs of vectors f1' and f2' in f1 and f2, computes 1 - (f1.f2),
    where '.' is the dot product, and rescales the results to lie in the
    range [cos_meas_min, cos_meas_max].
    Corresponds to regular cosine distance if f1' and f2' are normalized and
    cos_meas_min==0.0 and cos_meas_max==1.0."""
    return (1 - f1.T @ f2) * (cos_meas_max - cos_meas_min) + cos_meas_min


@jit(nopython=True)
def euclidean_distance(f1, f2, l2_meas_max=1.0, l2_meas_min=0.0):
    """Computes euclidean distances between the vectors in f1 and f2, and
    rescales the results to lie in the range [cos_meas_min, cos_meas_max]."""
    S1 = np.zeros((f1.shape[1], f2.shape[1]))
    for n in range(f2.shape[1]):
        S1[:, n] = np.sqrt(np.sum((f1.T - f2[:, n]) ** 2, axis=1))

    return S1 * (l2_meas_max - l2_meas_min) + l2_meas_min


def compute_high_res_cost_matrix(f_chroma1: np.ndarray,
                                 f_chroma2: np.ndarray,
                                 f_DLNCO1: np.ndarray,
                                 f_DLNCO2: np.ndarray,
                                 weights: np.ndarray = np.array([1.0, 1.0]),
                                 cos_meas_min: float = 1.0,
                                 cos_meas_max: float = 2.0,
                                 l2_meas_min: float = 0.0,
                                 l2_meas_max: float = 1.0):
    """Computes cost matrix of two sequences using two feature matrices
    for each sequence. Cosine distance is used for the chroma sequences and
    euclidean distance is used for the DLNCO sequences.

    Parameters
    ----------
    f_chroma1 : np.ndarray [shape=(12, N)]
        Chroma feature matrix of the first sequence (assumed to be normalized).

    f_chroma2 : np.ndarray [shape=(12, M)]
        Chroma feature matrix of the second sequence (assumed to be normalized).

    f_DLNCO1 : np.ndarray [shape=(12, N)]
        DLNCO feature matrix of the first sequence

    f_DLNCO2 : np.ndarray [shape=(12, M)]
        DLNCO feature matrix of the second sequence

    weights : np.ndarray [shape=[2,]]
        Weights array for the high-resolution cost computation.
        weights[0] * cosine_distance + weights[1] * euclidean_distance

    cos_meas_min : float
        Cosine distances are shifted to be at least ``cos_meas_min``

    cos_meas_max : float
        Cosine distances are scaled to be at most ``cos_meas_max``

    l2_meas_min : float
        Euclidean distances are shifted to be at least ``l2_meas_min``

    l2_meas_max : float
        Euclidean distances are scaled to be at most ``l2_meas_max``

    Returns
    -------
    C: np.ndarray [shape=(N, M)]
        Cost matrix
    """
    cos_dis = cosine_distance(f_chroma1, f_chroma2, cos_meas_min=cos_meas_min, cos_meas_max=cos_meas_max)
    euc_dis = euclidean_distance(f_DLNCO1, f_DLNCO2, l2_meas_min=l2_meas_min, l2_meas_max=l2_meas_max)

    return weights[0] * cos_dis + weights[1] * euc_dis

