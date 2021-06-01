from libfmp.c3 import compute_freq_distribution, tuning_similarity
from numba import jit
import numpy as np
from scipy import signal
from typing import Tuple


def smooth_downsample_feature(f_feature: np.ndarray,
                              input_feature_rate: float,
                              win_len_smooth: int = 0,
                              downsamp_smooth: int = 1) -> Tuple[np.ndarray, float]:
    """Temporal smoothing and downsampling of a feature sequence

    Parameters
    ----------
    f_feature : np.ndarray
        Input feature sequence, size dxN

    input_feature_rate : float
        Input feature rate in Hz

    win_len_smooth : int
        Smoothing window length. For 0, no smoothing is applied.

    downsamp_smooth : int
        Downsampling factor. For 1, no downsampling is applied.

    Returns
    -------
    f_feature_stat : np.array
        Downsampled & smoothed feature.

    new_feature_rate : float
        New feature rate after downsampling
    """
    if win_len_smooth != 0 or downsamp_smooth != 1:
        # hack to get the same results as on MATLAB
        stat_window = np.hanning(win_len_smooth+2)[1:-1]
        stat_window /= np.sum(stat_window)

        # upfirdn filters and downsamples each column of f_stat_help
        f_feature_stat = signal.upfirdn(h=stat_window, x=f_feature, up=1, down=downsamp_smooth)
        seg_num = f_feature.shape[1]
        stat_num = int(np.ceil(seg_num / downsamp_smooth))
        cut = int(np.floor((win_len_smooth - 1) / (2 * downsamp_smooth)))
        f_feature_stat = f_feature_stat[:, cut: stat_num + cut]
    else:
        f_feature_stat = f_feature

    new_feature_rate = input_feature_rate / downsamp_smooth

    return f_feature_stat, new_feature_rate


@jit(nopython=True)
def normalize_feature(feature: np.ndarray,
                      norm_ord: int,
                      threshold: float) -> np.ndarray:
    """Normalizes a feature sequence according to the l^norm_ord norm.

    Parameters
    ----------
    feature : np.ndarray
        Input feature sequence of size d x N
            d: dimensionality of feature vectors
            N: number of feature vectors (time in frames)

    norm_ord : int
        Norm degree

    threshold : float
        If the norm falls below threshold for a feature vector, then the
        normalized feature vector is set to be the normalized unit vector.

    Returns
    -------
    f_normalized : np.ndarray
        Normalized feature sequence
    """
    # TODO rewrite in vectorized fashion
    d, N = feature.shape
    f_normalized = np.zeros((d, N))

    # normalize the vectors according to the l^norm_ord norm
    unit_vec = np.ones(d)
    unit_vec = unit_vec / np.linalg.norm(unit_vec, norm_ord)

    for k in range(N):
        cur_norm = np.linalg.norm(feature[:, k], norm_ord)

        if cur_norm < threshold:
            f_normalized[:, k] = unit_vec
        else:
            f_normalized[:, k] = feature[:, k] / cur_norm

    return f_normalized


def estimate_tuning(x: np.ndarray,
                    Fs: float,
                    N: int = 16384,
                    gamma: float = 100,
                    local: bool = True,
                    filt: bool = True,
                    filt_len: int = 101) -> float:
    """Compute tuning deviation in cents for an audio signal. Convenience wrapper around
    'compute_freq_distribution' and 'tuning_similarity' from libfmp.

    Parameters
    ----------
    x : np.ndarray
        Input signal

    Fs : float
        Sampling rate

    N : int
        Window size

    gamma : float
        Constant for logarithmic compression

    local : bool
        If `True`, computes STFT and averages; otherwise computes global DFT

    filt : bool
        If `True`, applies local frequency averaging and by rectification

    filt_len : int
        Filter length for local frequency averaging (length given in cents)

    Returns
    -------
    tuning : float
        Estimated tuning deviation for ``x`` (in cents)
    """
    # TODO supply N in seconds and compute window size in frames via Fs
    v, _ = compute_freq_distribution(x, Fs, N, gamma, local, filt, filt_len)
    _, _, _, tuning, _ = tuning_similarity(v)
    return tuning


def shift_chroma_vectors(chroma: np.ndarray,
                         chroma_shift: int) -> np.ndarray:
    """Shift chroma representation by the given number of semitones.
    Format is assumed to be 12xN"""
    return np.roll(chroma, chroma_shift, axis=0)
