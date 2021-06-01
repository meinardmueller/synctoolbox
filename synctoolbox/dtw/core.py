import librosa
from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def __C_to_DE(C: np.ndarray = None,
              dn: np.ndarray = np.array([1, 1, 0], np.int64),
              dm: np.ndarray = np.array([1, 0, 1], np.int64),
              dw: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64),
              sub_sequence: bool = False) -> (np.ndarray, np.ndarray):
    """This function computes the accumulated cost matrix D and the step index
    matrix E.

    Parameters
    ----------
    C : np.ndarray (np.float32 / np.float64) [shape=(N, M)]
        Cost matrix

    dn : np.ndarray (np.int64) [shape=(1, S)]
        Integer array defining valid steps (N direction of C), default: [1, 1, 0]

    dm : np.ndarray (np.int64) [shape=(1, S)]
        Integer array defining valid steps (M direction of C), default: [1, 0, 1]

    dw : np.ndarray (np.float64) [shape=(1, S)]
        Double array defining the weight of the each step, default: [1.0, 1.0, 1.0]

    sub_sequence : bool
        Set `True` for SubSequence DTW, default: False

    Returns
    -------
    D : np.ndarray (np.float64) [shape=(N, M)]
        Accumulated cost matrix of type double

    E : np.ndarray (np.int64) [shape=(N, M)]
        Step index matrix.
        E[n, m] holds the index of the step take to determine the value of D[n, m].
        If E[n, m] is zero, no valid step was possible.
        NaNs in the cost matrix are preserved, invalid fields in the cost matrix are NaNs.
    """
    if C is None:
        raise ValueError('C must be a 2D numpy array.')

    N, M = C.shape
    S = dn.size

    if S != dm.size or S != dw.size:
        raise ValueError('The parameters dn,dm, and dw must be of equal length.')

    # calc bounding box size of steps
    sbbn = np.max(dn)
    sbbm = np.max(dm)

    # initialize E
    E = np.zeros((N, M), np.int64) - 1

    # initialize extended D matrix
    D = np.ones((sbbn + N, sbbm + M), np.float64) * np.inf

    if sub_sequence:
        for m in range(M):
            D[sbbn, sbbm + m] = C[0, m]
    else:
        D[sbbn, sbbm] = C[0, 0]

    # accumulate
    for m in range(sbbm, M + sbbm):
        for n in range(sbbn, N + sbbn):
            for s in range(S):
                cost = D[n - dn[s], m - dm[s]] + C[n - sbbn, m - sbbm] * dw[s]
                if cost < D[n, m]:
                    D[n, m] = cost
                    E[n - sbbn, m - sbbm] = s

    D = D[sbbn: N + sbbn, sbbm: M + sbbm]

    return D, E


@jit(nopython=True, cache=True)
def __E_to_warping_path(E: np.ndarray,
                        dn: np.ndarray = np.array([1, 1, 0], np.int64),
                        dm: np.ndarray = np.array([1, 0, 1], np.int64),
                        sub_sequence: bool = False,
                        end_index: int = -1) -> np.ndarray:
    """This function computes a warping path based on the provided matrix E
    and the allowed steps.

    Parameters
    ----------
    E : np.ndarray (np.int64) [shape=(N, M)]
        Step index matrix

    dn : np.ndarray (np.int64) [shape=(1, S)]
        Integer array defining valid steps (N direction of C), default: [1, 1, 0]

    dm : np.ndarray (np.int64) [shape=(1, S)]
         Integer array defining valid steps (M direction of C), default: [1, 0, 1]

    sub_sequence : bool
        Set `True` for SubSequence DTW, default: False

    end_index : int
        In case of SubSequence DTW

    Returns
    -------
    warping_path : np.ndarray (np.int64) [shape=(2, M)]
        Resulting optimal warping path
    """
    N, M = E.shape

    if not sub_sequence and end_index == -1:
        end_index = M - 1

    m = end_index
    n = N - 1

    warping_path = np.zeros((2, n + m + 1))

    index = 0

    def _loop(m, n, index):
        warping_path[:, index] = np.array([n, m])
        step_index = E[n, m]
        m -= dm[step_index]
        n -= dn[step_index]
        index += 1
        return m, n, index

    if sub_sequence:
        while n > 0:
            m, n, index = _loop(m, n, index)
    else:
        while m > 0 or n > 0:
            m, n, index = _loop(m, n, index)

    warping_path[:, index] = np.array([n, m])
    warping_path = warping_path[:, index::-1]

    return warping_path


def compute_warping_path(C: np.ndarray,
                         step_sizes: np.ndarray = np.array([[1, 0], [0, 1], [1, 1]], np.int64),
                         step_weights: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64),
                         implementation: str = 'synctoolbox'):
    """Applies DTW on cost matrix C.

    Parameters
    ----------
    C : np.ndarray (np.float32 / np.float64) [shape=(N, M)]
        Cost matrix

    step_sizes : np.ndarray (np.int64) [shape=(2, S)]
        Array of step sizes

    step_weights : np.ndarray (np.float64) [shape=(2, S)]
        Array of step weights

    implementation: str
        Choose among ``synctoolbox`` and ``librosa``. (default: ``synctoolbox``)

    Returns
    -------
    D : np.ndarray (np.float64) [shape=(N, M)]
        Accumulated cost matrix

    E : np.ndarray (np.int64) [shape=(N, M)]
        Step index matrix

    wp : np.ndarray (np.int64) [shape=(2, M)]
        Warping path
    """
    if implementation == 'librosa':
        D, wp, E = librosa.sequence.dtw(C=C,
                                        step_sizes_sigma=step_sizes,
                                        weights_add=np.array([0, 0, 0]),
                                        weights_mul=step_weights,
                                        return_steps=True,
                                        subseq=False)
        wp = wp[::-1].T

    elif implementation == 'synctoolbox':
        dn = step_sizes[:, 0]
        dm = step_sizes[:, 1]

        D, E = __C_to_DE(C,
                         dn=dn,
                         dm=dm,
                         dw=step_weights,
                         sub_sequence=False)

        wp = __E_to_warping_path(E=E,
                                 dn=dn,
                                 dm=dm,
                                 sub_sequence=False)

    else:
        raise NotImplementedError(f'No implementation found called {implementation}')

    return D, E, wp

