import librosa
import numpy as np

from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.csv_tools import read_csv_to_df, df_to_pitch_features, df_to_pitch_onset_features
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.filterbank import generate_filterbank
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features, __find_peaks
from synctoolbox.feature.utils import estimate_tuning
from utils import dict_allclose, filterbank_equal, load_dict


def test_tuning():
    import libfmp.c2  # TODO: This should be removed after the new release of libfmp
    audio_1, _ = librosa.load('data_music/Schubert_D911-03_HU33.wav', sr=22050)
    audio_2, _ = librosa.load('data_music/Schubert_D911-03_SC06.wav', sr=22050)
    tuning_offset_1 = estimate_tuning(audio_1, 22050)
    tuning_offset_2 = estimate_tuning(audio_2, 22050)

    assert tuning_offset_1 == 7
    assert tuning_offset_2 == 1


def test_chroma_features():
    f_pitch_1_gt = np.load('tests/data/f_pitch_1.npy')
    f_pitch_2_gt = np.load('tests/data/f_pitch_2.npy')
    f_chroma_1_gt = np.load('tests/data/f_chroma_1.npy')
    f_chroma_2_gt = np.load('tests/data/f_chroma_2.npy')

    f_chroma_1 = pitch_to_chroma(f_pitch=f_pitch_1_gt)
    f_chroma_2 = pitch_to_chroma(f_pitch=f_pitch_2_gt)

    assert np.allclose(f_chroma_1, f_chroma_1_gt, atol=1e-5)
    assert np.allclose(f_chroma_2, f_chroma_2_gt, atol=1e-5)


def test_quantized_chroma_features():
    f_chroma_1_gt = np.load('tests/data/f_chroma_1.npy')
    f_chroma_2_gt = np.load('tests/data/f_chroma_2.npy')
    f_chroma_quantized_1_gt = np.load('tests/data/f_chroma_quantized_1.npy')
    f_chroma_quantized_2_gt = np.load('tests/data/f_chroma_quantized_2.npy')

    f_chroma_quantized_1 = quantize_chroma(f_chroma=f_chroma_1_gt)
    f_chroma_quantized_2 = quantize_chroma(f_chroma=f_chroma_2_gt)

    assert np.allclose(f_chroma_quantized_1_gt, f_chroma_quantized_1, atol=1e-5)
    assert np.allclose(f_chroma_quantized_2_gt, f_chroma_quantized_2, atol=1e-5)


def test_CENS_features():
    f_chroma_quantized_1_gt = np.load('tests/data/f_chroma_quantized_1.npy')
    f_chroma_quantized_2_gt = np.load('tests/data/f_chroma_quantized_2.npy')
    f_CENS_1_gt = np.load('tests/data/f_CENS_1.npy')
    f_CENS_2_gt = np.load('tests/data/f_CENS_2.npy')

    f_cens_1 = quantized_chroma_to_CENS(f_chroma_quantized_1_gt, 201, 50, 50)[0]
    f_cens_2 = quantized_chroma_to_CENS(f_chroma_quantized_2_gt, 201, 50, 50)[0]

    assert np.allclose(f_cens_1, f_CENS_1_gt, atol=1e-5)
    assert np.allclose(f_cens_2, f_CENS_2_gt, atol=1e-5)


def test_filterbank():
    fb_gt = load_dict('tests/data/fb.pickle')
    fb = generate_filterbank(semitone_offset_cents=7)
    filterbank_equal(fb, fb_gt)


def test_peak_search():
    f_onset_gt = np.load('tests/data/f_onset.npy')
    peaks_gt = np.load('tests/data/peaks.npy')
    thresh_gt = np.load('tests/data/thresh.npy')
    time_peaks = __find_peaks(W=f_onset_gt, dir=1, abs_thresh=thresh_gt)
    assert np.array_equal(peaks_gt, time_peaks)


def test_pitch_features():
    audio_1, _ = librosa.load('data_music/Schubert_D911-03_HU33.wav', sr=22050)
    audio_2, _ = librosa.load('data_music/Schubert_D911-03_SC06.wav', sr=22050)
    f_pitch_1_gt = np.load('tests/data/f_pitch_1.npy')
    f_pitch_2_gt = np.load('tests/data/f_pitch_2.npy')

    f_pitch_1 = audio_to_pitch_features(f_audio=audio_1,
                                        Fs=22050,
                                        tuning_offset=1,
                                        feature_rate=50,
                                        verbose=False)

    f_pitch_2 = audio_to_pitch_features(f_audio=audio_2,
                                        Fs=22050,
                                        tuning_offset=7,
                                        feature_rate=50,
                                        verbose=False)

    assert np.allclose(f_pitch_1, f_pitch_1_gt, atol=1e-5)
    assert np.allclose(f_pitch_2, f_pitch_2_gt, atol=1e-5)


def test_pitch_onset_features():
    audio_1, _ = librosa.load('data_music/Schubert_D911-03_HU33.wav', sr=22050)
    audio_2, _ = librosa.load('data_music/Schubert_D911-03_SC06.wav', sr=22050)
    f_pitch_onset_1_gt = load_dict('tests/data/f_pitch_onset_1.pickle')
    f_pitch_onset_2_gt = load_dict('tests/data/f_pitch_onset_2.pickle')

    f_pitch_onset_1 = audio_to_pitch_onset_features(f_audio=audio_1,
                                                    Fs=22050,
                                                    tuning_offset=1,
                                                    verbose=False)

    f_pitch_onset_2 = audio_to_pitch_onset_features(f_audio=audio_2,
                                                    Fs=22050,
                                                    tuning_offset=7,
                                                    verbose=False)

    dict_allclose(f_pitch_onset_1, f_pitch_onset_1_gt, atol=1e-5)
    dict_allclose(f_pitch_onset_2, f_pitch_onset_2_gt, atol=1e-5)


def test_DLNCO_features():
    f_pitch_onset_1_gt = load_dict('tests/data/f_pitch_onset_1.pickle')
    f_pitch_onset_2_gt = load_dict('tests/data/f_pitch_onset_2.pickle')
    f_DLNCO_1_gt = np.load('tests/data/f_DLNCO_1.npy')
    f_DLNCO_2_gt = np.load('tests/data/f_DLNCO_2.npy')

    f_DLNCO_1 = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset_1_gt,
                                              feature_rate=50,
                                              feature_sequence_length=7518,
                                              visualize=False)

    f_DLNCO_2 = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset_2_gt,
                                              feature_rate=50,
                                              feature_sequence_length=6860,
                                              visualize=False)

    assert np.allclose(f_DLNCO_1, f_DLNCO_1_gt, atol=1e-5)
    assert np.allclose(f_DLNCO_2, f_DLNCO_2_gt, atol=1e-5)


def test_df_to_pitch_features():
    df_annotation = read_csv_to_df('data_csv/Chopin_Op010-03-Measures1-8_MIDI.csv', csv_delimiter=';')
    f_pitch_ann_gt = np.load('tests/data/f_pitch_ann.npy')

    f_pitch_ann = df_to_pitch_features(df_annotation, feature_rate=50)

    assert np.allclose(f_pitch_ann, f_pitch_ann_gt, atol=1e-5)


def test_df_to_pitch_onset_features():
    df_annotation = read_csv_to_df('data_csv/Chopin_Op010-03-Measures1-8_MIDI.csv', csv_delimiter=';')
    f_pitch_onset_ann_gt = load_dict('tests/data/f_pitch_onset_ann.pickle')

    f_pitch_onset_ann = df_to_pitch_onset_features(df_annotation)

    dict_allclose(f_pitch_onset_ann_gt, f_pitch_onset_ann, atol=1e-5)



