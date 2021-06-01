import numpy as np
from typing import Tuple

from synctoolbox.feature.utils import smooth_downsample_feature, normalize_feature


def pitch_to_CENS(f_pitch: np.ndarray,
                  input_feature_rate: float,
                  win_len_smooth: int = 0,
                  downsamp_smooth: int = 1,
                  quant_steps: np.ndarray = np.array([40, 20, 10, 5]) / 100,
                  quant_weights: np.ndarray = np.array([1, 1, 1, 1]) / 4,
                  norm_thresh: float = 0.001,
                  midi_min: int = 21,
                  midi_max: int = 108,
                  ) -> Tuple[np.ndarray, float]:
    """Generate CENS features from pitch features (CENS: Chroma Energy Normalized Statistics).

    The following is computed:

        * Energy for each chroma band

        * l1-normalization of the chroma vectors

        * Local statistics:

            + Component-wise quantization of the normalized chroma vectors

            + Smoothing and downsampling of the feature sequence

            + l2-normalization of the resulting vectors

    Individual steps of this procedure can be computed with the remaining functions in this module.

    Parameters
    ----------
    f_pitch : np.ndarray [shape=(128, N)]
        MIDI pitch-based feature representation, obtained e.g. through ``audio_to_pitch_features``.

    input_feature_rate: float
        Feature rate of the input pitch features ``f_pitch``

    win_len_smooth : int
        Smoothing window length, default: no smoothing

    downsamp_smooth : int
        Downsampling factor, default: no downsampling

    quant_steps : np.ndarray
        After l1-normalization, all entries are quantized into bins defined by these boundaries.
        The default values correspond to the standard definition of CENS features.

    quant_weights : np.ndarray
        The individual quantization bins can be given weights. Default is equal weight for all bins.

    norm_thresh : float
        For l1-normalization, chroma entries below this threshold are considered
        as noise and set to 0.
        For l2-normalization, chroma vectors with norm below this threshold
        are replaced with uniform vectors.

    midi_min : int
        Minimum MIDI pitch index to consider (default: 21)

    midi_max : int
        Maximum MIDI pitch index to consider (default: 108)

    Returns
    -------
    f_CENS: np.ndarray
        CENS (Chroma Energy Normalized Statistics) features

    CENS_feature_rate: float
        Feature rate of the CENS features
    """
    # Pitch to chroma features
    f_chroma = pitch_to_chroma(f_pitch=f_pitch,
                               midi_min=midi_min,
                               midi_max=midi_max)

    # Quantize chroma features
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma,
                                         quant_steps=quant_steps,
                                         quant_weights=quant_weights,
                                         norm_thresh=norm_thresh)

    # Temporal smoothing and downsampling
    f_CENS, CENS_feature_rate = quantized_chroma_to_CENS(f_chroma_quantized,
                                                         win_len_smooth,
                                                         downsamp_smooth,
                                                         input_feature_rate,
                                                         norm_thresh)

    return f_CENS, CENS_feature_rate


def quantized_chroma_to_CENS(f_chroma_quantized: np.ndarray,
                             win_len_smooth: int,
                             downsamp_smooth: int,
                             input_feature_rate: float,
                             norm_thresh: float = 0.001):
    """Smooths, downsamples, and normalizes a chroma sequence obtained e.g. through ``quantize_chroma``.

    Parameters
    ----------
    f_chroma_quantized: np.ndarray [shape=(12, N)]
        Quantized chroma representation

    win_len_smooth : int
        Smoothing window length. Setting this to 0 applies no smoothing.

    downsamp_smooth : int
        Downsampling factor. Setting this to 1 applies no downsampling.

    input_feature_rate: float
        Feature rate of ``f_chroma_quantized``

    norm_thresh : float
        For the final l2-normalization, chroma vectors with norm below this threshold
        are replaced with uniform vectors.

    Returns
    -------
    f_CENS: np.ndarray
        CENS (Chroma Energy Normalized Statistics) features

    CENS_feature_rate: float
        Feature rate of the CENS features
    """
    # Temporal smoothing and downsampling
    f_chroma_energy_stat, CENSfeature_rate = smooth_downsample_feature(f_feature=f_chroma_quantized,
                                                                       win_len_smooth=win_len_smooth,
                                                                       downsamp_smooth=downsamp_smooth,
                                                                       input_feature_rate=input_feature_rate)

    # Last step: normalize each vector with its L2 norm
    f_CENS = normalize_feature(feature=f_chroma_energy_stat, norm_ord=2, threshold=norm_thresh)

    return f_CENS, CENSfeature_rate


def quantize_chroma(f_chroma,
                    quant_steps: np.ndarray = np.array([40, 20, 10, 5]) / 100,
                    quant_weights: np.ndarray = np.array([1, 1, 1, 1]) / 4,
                    norm_thresh: float = 0.001) -> np.ndarray:
    """Computes thresholded l1-normalization of the chroma vectors and then applies
    component-wise quantization of the normalized chroma vectors.

    Parameters
    ----------
    f_chroma: np.ndarray [shape=(12, N)]
        Chroma representation

    quant_steps : np.ndarray
        After l1-normalization, all entries are quantized into bins defined by these boundaries.
        The default values correspond to the standard definition of CENS features.

    quant_weights : np.ndarray
        The individual quantization bins can be given weights. Default is equal weight for all bins.

    norm_thresh : float
        For l1-normalization, chroma entries below this threshold are considered
        as noise and set to 0.

    Returns
    -------
    f_chroma_quantized: np.ndarray [shape=(12, N)]
        Quantized chroma representation
    """
    f_chroma_energy_distr = np.zeros((12, f_chroma.shape[1]))

    # Thresholded l1-normalization
    for k in range(f_chroma.shape[1]):
        if np.sum(f_chroma[:, k] > norm_thresh) > 0:
            seg_energy_square = np.sum(f_chroma[:, k])
            f_chroma_energy_distr[:, k] = f_chroma[:, k] / seg_energy_square

    # component-wise quantization of the normalized chroma vectors
    f_chroma_quantized = np.zeros((12, f_chroma.shape[1]))
    for n in range(quant_steps.size):
        f_chroma_quantized += (f_chroma_energy_distr > quant_steps[n]) * quant_weights[n]

    return f_chroma_quantized


def pitch_to_chroma(f_pitch: np.ndarray,
                    midi_min: int = 21,
                    midi_max: int = 108) -> np.ndarray:
    """Aggregate pitch-based features into chroma bands.

    Parameters
    ----------
    f_pitch : np.ndarray [shape=(128, N)]
        MIDI pitch-based feature representation, obtained e.g. through
        ``audio_to_pitch_features``.

    midi_min : int
        Minimum MIDI pitch index to consider (default: 21)

    midi_max : int
        Maximum MIDI pitch index to consider (default: 108)

    Returns
    -------
    f_chroma: np.ndarray [shape=(12, N)]
        Rows of 'f_pitch' between ``midi_min`` and ``midi_max``,
        aggregated into chroma bands.
    """
    f_chroma = np.zeros((12, f_pitch.shape[1]))
    for p in range(midi_min, midi_max + 1):
        chroma = np.mod(p, 12)
        f_chroma[chroma, :] += f_pitch[p, :]
    return f_chroma
