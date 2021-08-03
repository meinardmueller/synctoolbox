import numpy as np

from synctoolbox.dtw.core import compute_warping_path
from synctoolbox.dtw.cost import cosine_distance
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import compute_optimal_chroma_shift, shift_chroma_vectors


def test_optimal_chroma_shift():
    f_CENS_1_gt = np.load('tests/data/f_CENS_1.npy')
    f_CENS_2_gt = np.load('tests/data/f_CENS_2.npy')
    opt_chroma_shift = compute_optimal_chroma_shift(f_CENS_1_gt, f_CENS_2_gt)

    assert opt_chroma_shift == 11


def test_dtw():
    f_chroma_1_gt = np.load('tests/data/f_chroma_1.npy')
    f_chroma_2_gt = np.load('tests/data/f_chroma_2.npy')
    C_cosine = cosine_distance(f_chroma_1_gt, f_chroma_2_gt)

    wp_gt = np.load('tests/data/wp.npy')

    _, _, wp = compute_warping_path(C_cosine)

    assert np.array_equal(wp_gt, wp)


def test_simple_mrmsdtw():
    f_chroma_quantized_1_gt = np.load('tests/data/f_chroma_quantized_1.npy')
    f_chroma_quantized_2_gt = np.load('tests/data/f_chroma_quantized_2.npy')
    wp_gt = np.load('tests/data/wp_simple.npy')

    f_chroma_quantized_2_gt = shift_chroma_vectors(f_chroma_quantized_2_gt, 11)

    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_quantized_1_gt,
                          f_chroma2=f_chroma_quantized_2_gt,
                          input_feature_rate=50,
                          step_weights=np.array([1.5, 1.5, 2.0]),
                          threshold_rec=10 ** 6,
                          verbose=False)

    assert np.array_equal(wp_gt, wp)


def test_high_res_mrmsdtw():
    f_chroma_quantized_1_gt = np.load('tests/data/f_chroma_quantized_1.npy')
    f_chroma_quantized_2_gt = np.load('tests/data/f_chroma_quantized_2.npy')
    f_DLNCO_1_gt = np.load('tests/data/f_DLNCO_1.npy')
    f_DLNCO_2_gt = np.load('tests/data/f_DLNCO_2.npy')
    wp_gt = np.load('tests/data/wp_high_res.npy')

    f_chroma_quantized_2_gt = shift_chroma_vectors(f_chroma_quantized_2_gt, 11)
    f_DLNCO_2_gt = shift_chroma_vectors(f_DLNCO_2_gt, 11)

    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_quantized_1_gt,
                          f_onset1=f_DLNCO_1_gt,
                          f_chroma2=f_chroma_quantized_2_gt,
                          f_onset2=f_DLNCO_2_gt,
                          input_feature_rate=50,
                          step_weights=np.array([1.5, 1.5, 2.0]),
                          threshold_rec=10 ** 6,
                          verbose=False)

    assert np.array_equal(wp_gt, wp)
