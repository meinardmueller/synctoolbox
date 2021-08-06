import numpy as np
from scipy import signal

FILTERBANK_SETTINGS = [
    {
        'fs': 22050,
        'midi_min': 96,
        'midi_max': 120
    },
    {
        'fs': 4410,
        'midi_min': 60,
        'midi_max': 95
    },
    {
        'fs': 882,
        'midi_min': 21,
        'midi_max': 59
    }
]

FS_PITCH = [22050, 4410, 882]


def generate_filterbank(semitone_offset_cents: float = 0.0,
                        Q: float = 25.0,
                        stop: float = 2.0,
                        Rp: float = 1.0,
                        Rs: float = 50.0) -> dict:
    """Generate a multi-rate filterbank corresponding to different MIDI pitches.
    Used to create the pitch features in ``audio_to_pitch_features`` and the pitch
    onset features in ``audio_to_pitch_onset_features``."""
    pass_rel = 1 / (2 * Q)
    stop_rel = pass_rel * stop
    shifted_midi_freq = __get_shifted_midi_frequencies(semitone_offset_cents)
    h = dict()
    for setting in FILTERBANK_SETTINGS:
        nyq = setting['fs'] / 2
        for midi_pitch in range(setting['midi_min'], setting['midi_max'] + 1):
            h[midi_pitch] = dict()
            pitch = shifted_midi_freq[midi_pitch]
            Wp = np.array([pitch - pass_rel * pitch, pitch + pass_rel * pitch], np.float64) / nyq
            Ws = np.array([pitch - stop_rel * pitch, pitch + stop_rel * pitch], np.float64) / nyq
            n, Wn = signal.ellipord(wp=Wp, ws=Ws, gpass=Rp, gstop=Rs)
            h[midi_pitch] = signal.ellip(N=n, rp=Rp, rs=Rs, Wn=Wn, output='sos', btype='bandpass')

    return h


def __get_shifted_midi_frequencies(semitone_offset_cents: float) -> np.ndarray:
    """Returns MIDI center frequencies, shifted by the given offset.

    Parameters
    ----------
    semitone_offset_cents : float
        Offset in cents

    Returns
    -------
    np.ndarray : Shifted MIDI center frequencies
    """
    return 2 ** ((np.arange(128) - 69 + semitone_offset_cents / 100) / 12) * 440.0


def generate_list_of_downsampled_audio(f_audio: np.ndarray) -> list:
    """Generates a multi resolution list of raw audio using downsampling

    Parameters
    ----------
    f_audio: np.ndarray
        Input audio array (mono)

    Returns
    -------
    wav_ds: list
        - wav_ds[0]: Same as ``f_audio``
        - wav_ds[1]: ``f_audio`` downsampled by the factor of 5, using a Kaiser window
        - wav_ds[2]: ``f_audio`` downsampled by the factor of 25, using a Kaiser window
    """
    wav_ds = list()
    wav_ds.append(f_audio)
    kaiser_win = __design_kaiser_win(up=1, down=5)
    wav_ds.append(signal.resample_poly(x=f_audio, up=1, down=5, axis=0, window=kaiser_win))
    wav_ds.append(signal.resample_poly(x=wav_ds[1], up=1, down=5, axis=0, window=kaiser_win))

    return wav_ds


def __design_kaiser_win(up: int, down: int, bta=5.0) -> np.ndarray:
    """This function is a workaround to have the same Kaiser window as in the
    resample() function in MATLAB."""
    max_rate = max(up, down)
    f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
    half_len = 100 * max_rate  # reasonable cutoff for our sinc-like function
    h = signal.firwin(2 * half_len + 1, f_c, window=('kaiser', bta))
    return h


def get_fs_index(midi_pitch: int) -> int:
    """Get the index of the filterbank used for `midi_pitch`"""
    if 21 <= midi_pitch <= 59:
        return 2
    elif 60 <= midi_pitch <= 95:
        return 1
    elif 96 <= midi_pitch <= 120:
        return 0
    else:
        raise ValueError('Invalid MIDI pitch {midi_pitch}! Choose between 21 <= midi_pitch <= 120.')