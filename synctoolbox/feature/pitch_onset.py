from libfmp.b import compressed_gray_cmap
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numba import jit

from synctoolbox.feature.filterbank import generate_list_of_downsampled_audio, get_fs_index,\
    generate_filterbank

EPS = np.finfo(float).eps

# Setting parameters #
######################
# Parameters for energy curves
# Fs = 22050, window size 101/22050: 4.6 ms
#             down_step: 50/22050: 2.3 ms
#             22050/50: 441,
# TODO:
#               samples ist kein guter Begriff hier,
#               weil überlagert mit dem PCM Sample Begriff
#
# Fs = 4410, window size 41/4410: 9.3 ms
#            down_step: 10/4410: 2.3 ms
#            4410/10: 441
#
# Fs = 882, window size 21/882: 23.8 ms
#            down_step: 10/882: 11.3 ms
#            882/10: 88.2

WINDOW_SIZE = np.array([101, 41, 21], int)
DOWNSAMPLING_FACTORS = np.array([50, 10, 10], int)
CUT_OFF = 1 / DOWNSAMPLING_FACTORS
CUT_OFF[2] = 0.05
RES_FAC = np.array([50, 50, 250])
WINDOW_LENGTHS = np.array([100, 100, 50], int)  # window lengths for local maximum


def audio_to_pitch_onset_features(f_audio: np.ndarray,
                                  Fs: float = 22050,
                                  midi_min: int = 21,
                                  midi_max: int = 108,
                                  tuning_offset: float = 0,
                                  manual_offset: float = -25,
                                  verbose: bool = False) -> dict:
    """Computes pitch onset features based on an IIR filterbank. The signal is decomposed
    into subbands that correspond to MIDI pitches between midi_min and midi_max.
    After that, onsets for each MIDI pitch are calculated.

    Parameters
    ----------
    f_audio : np.ndarray
        One dimensional audio array (mono)

    Fs : float
        Sampling rate of ``f_audio`` (in Hz)

    midi_min : int
        Minimum MIDI index (indices below ``midi_min`` are filled with zero in the output)

    midi_max : int
        Maximum MIDI index (indices above ``midi_max`` are filled with zero in the output)

    tuning_offset : float
        Tuning offset used to shift the filterbank (in cents)

    manual_offset : int
        Offset applied to all onsets (in ms). The procedure in this function finds
        onsets by looking at peaks, i.e., positions of maximum increase in energy.
        However, the actual onsets usually happen before such a peak (prior to the
        maximum increase in energy). Thus, an offset is applied to all onset
        positions. The default (-25ms) has been found to work well empirically.

    verbose : bool
        Set `True` to activate the visualization of features

    Returns
    -------
    f_peaks : dict
        A dictionary of onset peaks:
            * Each key corresponds to the midi pitch number
            * Each value f_peaks[midi_pitch] is an array of doubles of size 2xN:
                * First row give the positions of the peaks in milliseconds.
                * Second row contains the corresponding magnitudes of the peaks.
    """
    if verbose:
        print("Generating filterbank...")
    h = generate_filterbank(semitone_offset_cents=tuning_offset)

    if verbose:
        print("Downsampling signal...")
    wav_ds = generate_list_of_downsampled_audio(f_audio)

    # in peak picking for fs=22050, fs=4410, fs=882
    # Compute peak features for all pitches
    # (only pitches and harmonics occurring in the input MIDI are considered)
    # - due to memory requirements, we do note save f_filtfilt and f_onset
    # - threshold for peak picking is set to 2/3 of some local maximum
    # - all peaks are given with respect to fs = 22050

    f_peaks = dict()

    # Computing f_onset and f_peaks for all pitches
    if verbose:
        print("Processing midi pitches", midi_min, "to", midi_max)

    for midi_pitch in range(midi_min, midi_max + 1):
        if verbose and midi_pitch % 10 == 0:
            print(midi_pitch, end="")
        else:
            print(".", end="")
        index = get_fs_index(midi_pitch)

        f_filtfilt = signal.sosfiltfilt(x=wav_ds[index], sos=h[midi_pitch])
        ws = WINDOW_SIZE[index]
        ds = DOWNSAMPLING_FACTORS[index]

        # compute local energy curve
        f_square = f_filtfilt ** 2
        window = np.hanning(ws + 2)[1:-1]
        f_energy = signal.upfirdn(h=window, x=f_square, up=1, down=ds)
        delay = np.floor((ws-1)/(ds*2)).astype(int)
        length = np.floor(f_filtfilt.size / ds).astype(int)
        f_energy = f_energy[delay: delay + length]

        # further smoothing of the energy curve
        if CUT_OFF[index] < 0.01 or CUT_OFF[index] > 0.98:
            raise ValueError('Cut off frequency too small or too large!')

        Wp = CUT_OFF[index]
        n, Wn = signal.cheb2ord(wp=Wp, ws=Wp+0.01, gpass=1, gstop=20)

        sos = signal.cheby2(N=n, rs=20, Wn=Wn, output='sos')
        f_energy = signal.sosfiltfilt(x=f_energy, sos=sos)

        # discrete differentiation and half-wave rectifying
        f_onset = np.diff(f_energy)
        f_onset *= f_onset > 0

        # compute f_peaks using a local threshold method
        # normalization of samples wrt Fs=22050
        f_len = f_onset.size
        win_len = WINDOW_LENGTHS[index]
        sample_first = 1
        thresh = np.zeros(f_onset.shape, dtype=np.float64)

        while sample_first <= f_len:
            sample_last = np.minimum(sample_first + win_len - 1, f_len)
            win_max = np.max(f_onset[sample_first-1:sample_last])
            thr = 2 * win_max / 3
            thresh[sample_first-1:sample_last] = np.array([thr] * (sample_last-sample_first+1), np.float64)
            sample_first += win_len

        res = RES_FAC[index]
        res_center = np.ceil(res/2)
        time_peaks = __find_peaks(W=f_onset, dir=1, abs_thresh=thresh)
        val_peaks = f_onset[time_peaks.astype(int)]
        time_peaks = (time_peaks * res - res_center) * 1000 / Fs

        if manual_offset != 0:
            time_peaks += manual_offset
            non_negative_indices = np.argwhere(time_peaks > 0)
            time_peaks = time_peaks[non_negative_indices]
            val_peaks = val_peaks[non_negative_indices]

        f_peaks[midi_pitch] = np.array([time_peaks, val_peaks])

    if verbose:
        print("")
        from libfmp.b import plot_matrix
        # There are three time resolution: one per sampling rate and window step size. This is the
        # highest time resolution that appairs (in sec). A base sampling rate of
        # 22050 and downsampling of factor 5 twice is assumed.
        highest_time_res = np.min(DOWNSAMPLING_FACTORS / np.array([Fs, Fs/5, Fs/25], np.float64))
        time_grid_width = np.ceil(f_audio.size / Fs / highest_time_res).astype(int)
        num_pitches = midi_max - midi_min + 1

        imagedata = np.zeros((time_grid_width, num_pitches))

        for midi_pitch in range(midi_min, midi_max + 1):
            if midi_pitch not in f_peaks:
                raise ValueError(f'MIDI Pitch {midi_pitch} cannot be found in the input array f_peaks!')

            for k in range(f_peaks[midi_pitch].shape[1]):
                timecoord = np.minimum(__ms2imagecoord(f_peaks[midi_pitch][0, k], highest_time_res), time_grid_width)\
                    .astype(int)
                imagedata[np.maximum(1, np.arange(timecoord-3, timecoord+4)), midi_pitch-midi_min] =\
                    f_peaks[midi_pitch][1, k]

        timescale = np.arange(0, time_grid_width) * highest_time_res - highest_time_res / 2
        pitchscale = np.arange(midi_min, midi_max+1)
        fig, ax, im = plot_matrix(imagedata.T + EPS,
                                  figsize=(9, 9),
                                  colorbar_aspect=50,
                                  extent=[timescale[0], timescale[-1], midi_min, midi_max],
                                  ylabel='MIDI pitch',
                                  xlabel='Time (seconds)',
                                  title='Pitch Onset Features',
                                  cmap=compressed_gray_cmap(alpha=100))
        ax[0].set_yticks(pitchscale[::2])
        ax[0].set_yticklabels(pitchscale[::2], fontsize=10)
        plt.show()

    return f_peaks


def __ms2imagecoord(timems: float,
                    highest_time_res: float) -> float:
    """Round ``timems`` onto a discrete grid defined by 'highest_time_res'"""
    coord = np.round(timems / (highest_time_res * 1000))
    return coord


def __find_peaks(W: np.ndarray,
                 descent_thresh: np.ndarray = None,
                 dir: int = -1,
                 abs_thresh: np.ndarray = None,
                 rel_thresh: np.ndarray = None,
                 tmin: int = None,
                 tmax: int = None) -> np.ndarray:
    """This function finds the significant peaks in a given one-dimensional
    W array.

    Parameters
    ----------
    W : np.ndarray
        Signal to be searched for (positive) peaks

    descent_thresh : np.ndarray
        Descent threshold.
        During peak candidate verification, if a slope change
        from negative to positive slope occurs at sample i BEFORE the descent has
        exceeded rel_thresh(i), and if descent_thresh(i) has not been exceeded yet,
        the current peak candidate will be dropped. This situation corresponds to
        a secondary peak occuring shortly after the current candidate peak
        (which might lead to a higher peak value)!

    dir : int
        +1 for forward peak searching, -1 for backward peak searching (default=-1)

    abs_thresh : np.ndarray
        Absolute threshold signal, i.e. only __find_peaks satisfying W[i]>=abs_thresh[i]
        will be reported. abs_thresh must have the same number of samples as W.
        A sensible choice for this parameter would be a global or local average
        or median of the signal W. If omitted, half the median of W will be used.

    rel_thresh : np.ndarray
        Relative threshold signal. Only peak positions i with an uninterrupted
        positive ascent before position i of at least rel_thresh[i] and a
        possibly interrupted (see parameter descent_thresh) descent of at least
        rel_thresh[i] will be reported. rel_thresh must have the same number of
        samples as W. A sensible choice would be some measure related to the
        global or local variance of the signal W. If omitted, half the
        standard deviation of W will be used.

    tmin : int
        Index of start sample. Peak search will begin at W[tmin]

    tmax : int
        Index of end sample. Peak search will end at W[tmax]

    Returns
    -------
    peaks : np.ndarray
        Column vector of peak positions
    """
    # Initialize parameters
    tmax = W.size if tmax is None else tmax
    tmin = 0 if tmin is None else tmin

    if dir == 1:
        range = np.arange(tmin, tmax - 1)
    elif dir == -1:
        range = np.arange(tmax - 1, tmin, -1)
    else:
        raise ValueError('Direction has to be either +1 or -1.')

    abs_thresh = np.repeat(0.5*np.median(W), W.size) if abs_thresh is None else abs_thresh
    rel_thresh = 0.5 * np.repeat(np.sqrt(np.var(W)), W.size) if rel_thresh is None else rel_thresh
    descent_thresh = 0.5 * rel_thresh if descent_thresh is None else descent_thresh

    peaks = __find_peaks_jit_helper(W, abs_thresh, descent_thresh, dir, range, rel_thresh)
    return peaks


@jit(nopython=True)
def __find_peaks_jit_helper(W, abs_thresh, descent_thresh, dir, range, rel_thresh):
    dyold = 0
    rise = 0  # current amount of ascent during a rising portion of the signal W
    riseold = 0  # accumulated amount of ascent from the last rising portion of W
    descent = 0  # current amount of descent (<0) during a falling portion of the signal W
    searching_peak = True
    candidate = 0
    # Initialization for the output __find_peaks array
    peaks_list = list()
    for i in range:
        dy = W[i + dir] - W[i]
        if dy >= 0:
            rise += dy
        else:
            descent += dy

        if dyold >= 0:
            if dy < 0:  # slope change positive -> negative
                if rise >= rel_thresh[i] and searching_peak:
                    candidate = i
                    searching_peak = False
                riseold = rise
                rise = 0
        else:  # dyold < 0
            if dy < 0:  # in descent
                if descent <= -rel_thresh[candidate] and not searching_peak:
                    if W[candidate] >= abs_thresh[candidate]:
                        peaks_list.append(candidate)
                    searching_peak = True
            else:  # dy >= 0 slope change negative->positive
                if not searching_peak:
                    if W[candidate] - W[i] <= descent_thresh[i]:
                        rise = riseold + descent
                    if descent <= -rel_thresh[candidate]:
                        if W[candidate] >= abs_thresh[candidate]:

                            peaks_list.append(candidate)
                    searching_peak = True
                descent = 0
        dyold = dy
    peaks = np.array(peaks_list, np.float64)
    return peaks
