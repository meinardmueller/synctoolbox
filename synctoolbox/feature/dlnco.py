import matplotlib.pyplot as plt
import numpy as np
from libfmp.b import MultiplePlotsWithColorbar, plot_chromagram, plot_matrix


def pitch_onset_features_to_DLNCO(f_peaks: dict,
                                  feature_sequence_length: int,
                                  feature_rate: int = 50,
                                  midi_min: int = 21,
                                  midi_max: int = 108,
                                  log_compression_gamma: float = 10000.0,
                                  chroma_norm_ord: int = 2,
                                  LN_maxfilterlength_seconds: float = 0.8,
                                  LN_maxfilterthresh: float = 0.1,
                                  DLNCO_filtercoef: np.ndarray = np.sqrt(1 / np.arange(1, 11)),
                                  visualize=False) -> np.ndarray:
    """Computes decaying locally adaptive normalized chroma onset (DLNCO) features from
    a dictionary of peaks obtained e.g. by ``audio_to_pitch_onset_features``.

    Parameters
    ----------
    f_peaks : dict
        A dictionary of onset peaks

            * Each key corresponds to the midi pitch number

            * Each value f_peaks[midi_pitch] is an array of doubles of size 2xN:

                + First row give the positions of the peaks in milliseconds.

                + Second row contains the corresponding magnitudes of the peaks.

    feature_sequence_length : int
        Desired length of the resulting feature sequence. This should be at least as long as the
        position of the last peak in ``f_peaks``, but can be longer.

    feature_rate : int
        Desired features per second in the output representation

    midi_min : int
        Minimum MIDI pitch index (default: 21)

    midi_max : int
        Maximum MIDI pitch index (default: 108)

    log_compression_gamma : float
        Gamma factor of the log compression applied to peak magnitudes.
        
    chroma_norm_ord : int
        Order of the norm used for chroma onset vectors.

    LN_maxfilterlength_seconds : float
        Length of the maximum filter applied for determining local norm of chroma onsets in seconds.

    LN_maxfilterthresh : float
        Minimum threshold for normalizing chroma onsets using local norm.

    DLNCO_filtercoef : float
        Sequence of decay coefficients applied on normalized chroma onsets.

    visualize : bool
        Set `True` to visualize chroma onset features (Default: False)

    Returns
    -------
    f_DLNCO : np.array [shape=(d_dlnco, N_dlnco)]
        Decaying Locally adaptively Normalized Chroma Onset features
    """
    f_CO = np.zeros((feature_sequence_length, 12))

    for midi_pitch in range(midi_min, midi_max + 1):
        if midi_pitch not in f_peaks:
            continue
        time_peaks = f_peaks[midi_pitch][0, :] / 1000  # Now given in seconds
        val_peaks = np.log(f_peaks[midi_pitch][1, :] * log_compression_gamma + 1)
        ind_chroma = np.mod(midi_pitch, 12)
        for k in range(time_peaks.size):
            indTime = __matlab_round(time_peaks[k] * feature_rate)  # Usage of "round" accounts
                                                         # "center window convention"

            f_CO[indTime, ind_chroma] += val_peaks[k]

    # No two ways to normalize F_CO: simply columnwise (f_N) or via local
    # normalizing curve (f_LN)
    f_N = np.zeros(feature_sequence_length)

    for k in range(feature_sequence_length):
        f_N[k] = np.linalg.norm(f_CO[k, :], chroma_norm_ord)

    f_LN = np.array(f_N, copy=True)
    f_left = np.array(f_N, copy=True)
    f_right = np.array(f_N, copy=True)
    LN_maxfilterlength_frames = int(LN_maxfilterlength_seconds * feature_rate)
    if LN_maxfilterlength_frames % 2 == 1:
        LN_maxfilterlength_frames -= 1
    shift = int(np.floor((LN_maxfilterlength_frames) / 2))

    # TODO improve with scipy.ndimage.maximum_filter
    for s in range(shift):
        f_left = np.roll(f_left, 1, axis=0)
        f_left[0] = 0
        f_right = np.roll(f_right, -1, axis=0)
        f_right[-1] = 0
        f_LN = np.max([f_left, f_LN, f_right], axis=0)

    f_LN = np.maximum(f_LN, LN_maxfilterthresh)

    # Compute f_NC0 (normalizing f_C0 using f_N)
    # f_NCO = np.zeros((feature_sequence_length, 12))

    # Compute f_LNC0 (normalizing f_C0 using f_LN)
    f_LNCO = np.zeros((feature_sequence_length, 12))
    for k in range(feature_sequence_length):
        # f_NCO[k, :] = f_CO[k, :] / (f_N[k]) #+ eps)
        f_LNCO[k, :] = f_CO[k, :] / f_LN[k]

    # Compute f_DLNCO
    f_DLNCO = np.zeros((feature_sequence_length, 12))

    num_coef = DLNCO_filtercoef.size
    for p_idx in range(12):
        v_shift = np.array(f_LNCO[:, p_idx], copy=True)
        v_help = np.zeros((feature_sequence_length, num_coef))

        for n in range(num_coef):
            v_help[:, n] = DLNCO_filtercoef[n] * v_shift
            v_shift = np.roll(v_shift, 1)
            v_shift[0] = 0

        f_DLNCO[:, p_idx] = np.max(v_help, axis=1)

    # visualization
    if visualize:
        plot_chromagram(X=f_CO.T, title='CO', colorbar=True, Fs=feature_rate, colorbar_aspect=50, figsize=(9, 3))
        __visualize_LN_features(f_N, f_LN, feature_sequence_length, feature_rate)
        plot_chromagram(X=f_LNCO.T, title='LNCO', colorbar=True, Fs=feature_rate, colorbar_aspect=50, figsize=(9, 3))
        plot_chromagram(X=f_DLNCO.T, title='DLNCO', colorbar=True, Fs=feature_rate, colorbar_aspect=50, figsize=(9, 3))

    f_DLNCO = f_DLNCO.T

    return f_DLNCO


def __visualize_LN_features(f_N: np.ndarray,
                            f_LN: np.ndarray,
                            num_feature: int,
                            res: int,
                            ax: plt.Axes = None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 3), dpi=72)

    t = np.arange(0, num_feature) / res
    ax.plot(t, f_N)

    if t[-1] > t[0]:
        ax.set_xlim([t[0], t[-1]])

    ax.plot(t, f_LN, 'r')
    ax.set_title('Local Norm of CO')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Norm')


def __matlab_round(x: float = None) -> int:
    """Workaround to cope the rounding differences between MATLAB and python"""
    if x - np.floor(x) < 0.5:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))
