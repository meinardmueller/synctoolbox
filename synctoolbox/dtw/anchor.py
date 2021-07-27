from numba import jit
import numpy as np
from typing import Tuple


def project_alignment_on_a_new_feature_rate(alignment: np.ndarray,
                                            feature_rate_old: int,
                                            feature_rate_new: int,
                                            cost_matrix_size_old: tuple = (),
                                            cost_matrix_size_new: tuple = ()) -> np.ndarray:
    """Projects an alignment computed for a cost matrix on a certain
    feature resolution on a cost matrix having a different feature
    resolution.

    Parameters
    ----------
    alignment : np.ndarray [shape=(2, N)]
        Alignment matrix

    feature_rate_old : int
        Feature rate of the old cost matrix

    feature_rate_new : int
        Feature rate of the new cost matrix

    cost_matrix_size_old : tuple
        Size of the old cost matrix. Possibly needed to deal with border cases

    cost_matrix_size_new : tuple
        Size of the new cost matrix. Possibly needed to deal with border cases

    Returns
    -------
    np.ndarray [shape=(2, N)]
        Anchor sequence for the new cost matrix
    """
    # Project the alignment on the new feature rate
    fac = feature_rate_new / feature_rate_old
    anchors = np.round(alignment * fac) + 1

    # In case the sizes of the cost matrices are given explicitly and the
    # alignment specifies to align the first and last elements, handle this case
    # separately since this might cause problems in the general projection
    # procedure.
    if cost_matrix_size_old is not None and cost_matrix_size_new is not None:
        if np.array_equal(alignment[:, 0], np.array([0, 0])):
            anchors[:, 0] = np.array([1, 1])

        if np.array_equal(alignment[:, -1], np.array(cost_matrix_size_old) - 1):
            anchors[:, -1] = np.array(cost_matrix_size_new)

    return anchors - 1


def derive_anchors_from_projected_alignment(projected_alignment: np.ndarray,
                                            threshold: int) -> np.ndarray:
    """Derive anchors from a projected alignment such that the area of the rectangle
    defined by two subsequent anchors a1 and a2 is below a given threshold.

    Parameters
    ----------
    projected_alignment : np.ndarray [shape=(2, N)]
        Projected alignment array

    threshold : int
        Maximum area of the constraint rectangle

    Returns
    -------
    anchors_res : np.ndarray [shape=(2, M)]
        Resulting anchor sequence
    """
    L = projected_alignment.shape[1]

    a1 = np.array(projected_alignment[:, 0], copy=True).reshape(-1, 1)
    a2 = np.array(projected_alignment[:, -1], copy=True).reshape(-1, 1)

    if __compute_area(a1, a2) <= threshold:
        anchors_res = np.concatenate([a1, a2], axis=1)
    elif L > 2:
        center = int(np.floor(L/2 + 1))

        a1 = np.array(projected_alignment[:, 0], copy=True).reshape(-1, 1)
        a2 = np.array(projected_alignment[:, center - 1], copy=True).reshape(-1, 1)
        a3 = np.array(projected_alignment[:, -1], copy=True).reshape(-1, 1)

        if __compute_area(a1, a2) > threshold:
            anchors_1 = derive_anchors_from_projected_alignment(projected_alignment[:, 0:center], threshold)
        else:
            anchors_1 = np.concatenate([a1, a2], axis=1)

        if __compute_area(a2, a3) > threshold:
            anchors_2 = derive_anchors_from_projected_alignment(projected_alignment[:, center - 1:], threshold)
        else:
            anchors_2 = np.concatenate([a2, a3], axis=1)

        anchors_res = np.concatenate([anchors_1, anchors_2[:, 1:]], axis=1)
    else:
        if __compute_area(a1, a2) > threshold:
            print('Only two anchor points are given which do not fulfill the constraint.')
        anchors_res = np.concatenate([a1, a2], axis=1)

    return anchors_res


def derive_neighboring_anchors(warping_path: np.ndarray,
                               anchor_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute anchor points in the neighborhood of previous anchor points.

    Parameters
    ----------
    warping_path : np.ndarray [shape=(2, N)]
        Warping path

    anchor_indices : np.ndarray
        Indices corresponding to the anchor points in the ``warping_path``

    Returns
    -------
    neighboring_anchors : np.ndarray [shape=(2, N-1)]
        Sequence of neighboring anchors

    neighboring_anchor_indices : np.ndarray
        Indices into ``warping path`` corresponding to ``neighboring_anchors``
    """
    L = anchor_indices.shape[0]
    neighboring_anchor_indices = np.zeros(L-1, dtype=int)
    neighboring_anchors = np.zeros((2, L-1),  dtype=int)

    for k in range(1, L):
        i1 = anchor_indices[k-1]
        i2 = anchor_indices[k]

        neighboring_anchor_indices[k-1] = i1 + np.floor((i2 - i1) / 2)
        neighboring_anchors[:, k-1] = warping_path[:, neighboring_anchor_indices[k - 1]]

    return neighboring_anchors, neighboring_anchor_indices


@jit(nopython=True)
def __compute_area(a: tuple,
                   b: tuple):
    """Computes the area between two points, given as tuples"""
    return (b[0] - a[0] + 1) * (b[1] - a[1] + 1)



