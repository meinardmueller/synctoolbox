from libfmp.b import plot_matrix
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy import signal

from synctoolbox.feature.filterbank import FS_PITCH, generate_list_of_downsampled_audio, get_fs_index,\
    generate_filterbank

PITCH_NAME_LABELS = ['   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C0 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C1 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C2 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C3 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C4 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C5 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C6 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C7 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C8 ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ', '   ',
                     'C9 ']


def audio_to_pitch_features(f_audio: np.ndarray,
                            Fs: float = 22050,
                            feature_rate: int = 50,
                            midi_min: int = 21,
                            midi_max: int = 108,
                            tuning_offset: int = 0,
                            verbose: bool = False) -> np.ndarray:
    """Computes pitch-based features via an IIR filterbank aggregated as STMSP
    (short-time mean-square power). The signal is decomposed into subbands that
    correspond to MIDI pitches between midi_min and midi_max.
    In the output array, each row corresponds to one MIDI pitch. Per convention,
    the output has size 128xN. Only the rows between ``midi_min`` and ``midi_max``
    are filled, the rest contains zeros.

    Parameters
    ----------
    f_audio : np.ndarray
        One dimensional audio array (mono)

    Fs : float
        Sampling rate of ``f_audio`` (in Hz)

    feature_rate: int
        Features per second

    midi_min : int
        Minimum MIDI index (indices below ``midi_min`` are filled with zero in the output)

    midi_max : int
        Maximum MIDI index (indices above ``midi_max`` are filled with zero in the output)

    tuning_offset : int
        Tuning offset used to shift the filterbank (in cents)

    verbose : bool
        Set `True` to activate the visualization of features

    Returns
    -------
    f_pitch : np.ndarray [shape=(128, N)]
        Matrix containing the extracted pitch-based features
    """
    if verbose:
        print("Generating filterbank...")
    h = generate_filterbank(semitone_offset_cents=tuning_offset)

    if verbose:
        print("Downsampling signal...")
    wav_ds = generate_list_of_downsampled_audio(f_audio)

    # Compute features for all pitches
    wav_size = f_audio.size
    win_len_STMSP = Fs / feature_rate * 2
    step_size = int(win_len_STMSP / 2)
    group_delay = np.round(win_len_STMSP / 2)

    # Group delay is adjusted
    seg_wav_start = np.concatenate([np.ones(1), np.arange(1, wav_size+1, step_size)]).astype(np.float64)
    seg_wav_stop = np.minimum(seg_wav_start + win_len_STMSP, wav_size)
    seg_wav_stop[0] = np.minimum(group_delay, wav_size)
    seg_wav_num = seg_wav_start.size
    f_pitch = np.zeros((128, seg_wav_num))

    if verbose:
        print("Processing midi pitches", midi_min, "to", midi_max)
    for midi_pitch in range(midi_min, midi_max + 1):
        if verbose and midi_pitch % 10 == 0:
            print(midi_pitch, end="")
        else:
            print(".", end="")
        index = get_fs_index(midi_pitch)
        f_filtfilt = signal.sosfiltfilt(x=wav_ds[index], sos=h[midi_pitch])
        f_square = f_filtfilt ** 2

        start = np.floor(seg_wav_start / Fs * FS_PITCH[index]).astype(int)  # floor due to indexing
        stop = np.floor(seg_wav_stop / Fs * FS_PITCH[index]).astype(int)
        factor = Fs / FS_PITCH[index]
        __window_and_sum(f_pitch, f_square, midi_pitch, seg_wav_num, start, stop, factor)

    if verbose:
        print("")
        __visualize_pitch(f_pitch, feature_rate=feature_rate)
        plt.show()

    return f_pitch


@jit(nopython=True)
def __window_and_sum(f_pitch, f_square, midi_pitch, seg_wav_num, start, stop, factor):
    for k in range(seg_wav_num):  # TODO this is extremely inefficient, can we use better numpy indexing to improve this? np.convolve?
        f_pitch[midi_pitch, k] = np.sum(f_square[start[k]:stop[k]]) * factor


def __visualize_pitch(f_pitch: np.ndarray,
                      midi_min: int = 21,
                      midi_max: int = 108,
                      feature_rate: float = 0,
                      use_pitch_name_labels: bool = False,
                      y_tick: np.ndarray = np.array([21, 30, 40, 50, 60, 70, 80, 90, 100], int)):

    f_image = f_pitch[midi_min:midi_max + 1, :]

    fig, ax, im = plot_matrix(X=f_image, extent=[0, f_pitch.shape[1]/feature_rate, midi_min, midi_max+1],
                              title='Pitch Features', ylabel='MIDI Pitch', figsize=(9, 9),
                              colorbar_aspect=50)

    pitchscale = np.arange(midi_min, midi_max + 1)

    ax[0].set_yticks(pitchscale[::2])
    if use_pitch_name_labels:
        ax[0].set_yticks(np.arange(midi_min, midi_max + 1))
        ax[0].set_yticklabels(PITCH_NAME_LABELS[midi_min-1:midi_max], fontsize=12)
    else:
        ax[0].set_yticks(pitchscale[::2])
        ax[0].set_yticklabels(pitchscale[::2], fontsize=10)

