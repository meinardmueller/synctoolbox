import numpy as np
import time

from synctoolbox.dtw.anchor import derive_anchors_from_projected_alignment, derive_neighboring_anchors, \
    project_alignment_on_a_new_feature_rate
from synctoolbox.dtw.utils import build_path_from_warping_paths, compute_cost_matrices_between_anchors, \
    compute_warping_paths_from_cost_matrices, find_anchor_indices_in_warping_path
from synctoolbox.dtw.visualization import sync_visualize_step1, sync_visualize_step2
from synctoolbox.feature.utils import smooth_downsample_feature, normalize_feature


def sync_via_mrmsdtw(f_chroma1: np.ndarray,
                     f_chroma2: np.ndarray,
                     f_DLNCO1: np.ndarray = None,
                     f_DLNCO2: np.ndarray = None,
                     input_feature_rate: float = 50,
                     step_sizes: np.ndarray = np.array([[1, 0], [0, 1], [1, 1]], np.int32),
                     step_weights: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64),
                     threshold_rec: int = 10000, win_len_smooth: np.ndarray = np.array([201, 101, 21, 1]),
                     downsamp_smooth: np.ndarray = np.array([50, 25, 5, 1]),
                     verbose: bool = False,
                     dtw_implementation: str = 'synctoolbox',
                     normalize_chroma: bool = True,
                     chroma_norm_ord: int = 2,
                     chroma_norm_threshold: float = 0.001):
    """Compute memory-restricted multi-scale DTW (MrMsDTW) using chroma and (optionally) DLNCO features.
    MrMsDTW is performed on multiple levels that get progressively finer, with rectangular constraint
    regions defined by the alignment found on the previous, coarser level.
    If DLNCO features are provided, these are used on the finest level in addition to chroma
    to provide higher synchronization accuracy.

    Parameters
    ----------
    f_chroma1 : np.ndarray [shape=(12, N)]
        Chroma feature matrix of the first sequence

    f_chroma2 : np.ndarray [shape=(12, M)]
        Chroma feature matrix of the second sequence

    f_DLNCO1 : np.ndarray [shape=(12, N)]
        DLNCO feature matrix of the first sequence (optional, default: None)

    f_DLNCO2 : np.ndarray [shape=(12, M)]
        DLNCO feature matrix of the second sequence (optional, default: None)

    input_feature_rate: float
        Input feature rate of the chroma features (default: 50)

    step_sizes: np.ndarray
        DTW step sizes (default: np.array([[1, 0], [0, 1], [1, 1]]))

    step_weights: np.ndarray
        DTW step weights (np.array([1.0, 1.0, 1.0]))

    threshold_rec: int
        Defines the maximum area that is spanned by the rectangle of two
        consecutive elements in the alignment (default: 10000)

    win_len_smooth : np.ndarray
        Window lengths for chroma feature smoothing (default: np.array([201, 101, 21, 1]))

    downsamp_smooth : np.ndarray
        Downsampling factors (default: np.array([50, 25, 5, 1]))

    verbose : bool
        Set `True` for visualization (default: False)

    dtw_implementation : str
        DTW implementation, librosa or synctoolbox (default: synctoolbox)

    normalize_chroma : bool
        Set `True` to normalize input chroma features after each downsampling
        and smoothing operation.

    chroma_norm_ord: int
        Order of chroma normalization, relevant if ``normalize_chroma`` is True.
        (default: 2)

    chroma_norm_threshold: float
        If the norm falls below threshold for a feature vector, then the
        normalized feature vector is set to be the unit vector. Relevant, if
        ``normalize_chroma`` is True (default: 0.001)

    Returns
    -------
    alignment : np.ndarray [shape=(2, T)]
        Resulting warping path
    """
    # If DLNCO features are given as input, high resolution MrMsDTW is activated.
    high_res = False
    if f_DLNCO1 is not None and f_DLNCO2 is not None:
        high_res = True

    if high_res and (f_chroma1.shape[1] != f_DLNCO1.shape[1] or f_chroma2.shape[1] != f_DLNCO2.shape[1]):
        raise ValueError('Chroma and DLNCO features must be of the same length.')

    if downsamp_smooth[-1] != 1 or win_len_smooth[-1] != 1:
        raise ValueError('The downsampling factor of the last iteration must be equal to 1, i.e.'
                         'at the last iteration, it is computed at the input feature rate!')

    num_iterations = win_len_smooth.shape[0]
    cost_matrix_size_old = tuple()
    feature_rate_old = input_feature_rate / downsamp_smooth[0]
    alignment = None

    total_computation_time = 0.0
    for it in range(num_iterations):
        tic1 = time.perf_counter()

        # Smooth and downsample given raw features
        f_chroma1_cur, _ = smooth_downsample_feature(f_chroma1,
                                                     input_feature_rate=input_feature_rate,
                                                     win_len_smooth=win_len_smooth[it],
                                                     downsamp_smooth=downsamp_smooth[it])

        f_chroma2_cur, feature_rate_new = smooth_downsample_feature(f_chroma2,
                                                                    input_feature_rate=input_feature_rate,
                                                                    win_len_smooth=win_len_smooth[it],
                                                                    downsamp_smooth=downsamp_smooth[it])

        if normalize_chroma:
            f_chroma1_cur = normalize_feature(f_chroma1_cur,
                                              norm_ord=chroma_norm_ord,
                                              threshold=chroma_norm_threshold)

            f_chroma2_cur = normalize_feature(f_chroma2_cur,
                                              norm_ord=chroma_norm_ord,
                                              threshold=chroma_norm_threshold)

        # Project path onto new resolution
        cost_matrix_size_new = (f_chroma1_cur.shape[1], f_chroma2_cur.shape[1])

        if alignment is None:
            # Initialize the alignment with the start and end frames of the feature sequence
            anchors = np.array([[0, f_chroma1_cur.shape[1] - 1], [0, f_chroma2_cur.shape[1] - 1]])

        else:
            projected_alignment = project_alignment_on_a_new_feature_rate(alignment=alignment,
                                                                          feature_rate_old=feature_rate_old,
                                                                          feature_rate_new=feature_rate_new,
                                                                          cost_matrix_size_old=cost_matrix_size_old,
                                                                          cost_matrix_size_new=cost_matrix_size_new)

            anchors = derive_anchors_from_projected_alignment(projected_alignment=projected_alignment,
                                                              threshold=threshold_rec)

        # Cost matrix and warping path computation
        if high_res and it == num_iterations - 1:
            # Compute cost considering chroma and pitch onset features and alignment only in the last iteration,
            # where the features are at the finest level.
            cost_matrices_step1 = compute_cost_matrices_between_anchors(f_chroma1=f_chroma1_cur,
                                                                        f_chroma2=f_chroma2_cur,
                                                                        f_DLNCO1=f_DLNCO1,
                                                                        f_DLNCO2=f_DLNCO2,
                                                                        anchors=anchors)

        else:
            cost_matrices_step1 = compute_cost_matrices_between_anchors(f_chroma1=f_chroma1_cur,
                                                                        f_chroma2=f_chroma2_cur,
                                                                        anchors=anchors)

        wp_list = compute_warping_paths_from_cost_matrices(cost_matrices_step1,
                                                           step_sizes=step_sizes,
                                                           step_weights=step_weights,
                                                           implementation=dtw_implementation)

        # Concatenate warping paths
        wp = build_path_from_warping_paths(warping_paths=wp_list,
                                           anchors=anchors)

        anchors_step1 = None
        wp_step1 = None
        num_rows_step1 = 0
        num_cols_step1 = 0
        ax = None

        toc1 = time.perf_counter()
        if verbose and cost_matrices_step1 is not None:
            anchors_step1 = np.array(anchors, copy=True)
            wp_step1 = np.array(wp, copy=True)
            num_rows_step1, num_cols_step1 = np.sum(np.array([dtw_mat.shape for dtw_mat in cost_matrices_step1], int),
                                                    axis=0)
            fig, ax = sync_visualize_step1(cost_matrices_step1,
                                           num_rows_step1,
                                           num_cols_step1,
                                           anchors,
                                           wp)
        tic2 = time.perf_counter()

        # Compute neighboring anchors and refine alignment using local path between neighboring anchors
        anchor_indices_in_warping_path = find_anchor_indices_in_warping_path(wp, anchors=anchors)

        # Compute neighboring anchors for refinement
        neighboring_anchors, neighboring_anchor_indices = \
            derive_neighboring_anchors(wp, anchor_indices=anchor_indices_in_warping_path)

        if neighboring_anchor_indices.shape[0] > 1 \
                and it == num_iterations - 1 and high_res:
            cost_matrices_step2 = compute_cost_matrices_between_anchors(f_chroma1=f_chroma1_cur,
                                                                        f_chroma2=f_chroma2_cur,
                                                                        f_DLNCO1=f_DLNCO1,
                                                                        f_DLNCO2=f_DLNCO2,
                                                                        anchors=neighboring_anchors)

        else:
            cost_matrices_step2 = compute_cost_matrices_between_anchors(f_chroma1=f_chroma1_cur,
                                                                        f_chroma2=f_chroma2_cur,
                                                                        anchors=neighboring_anchors)

        wp_list_refine = compute_warping_paths_from_cost_matrices(cost_matrices=cost_matrices_step2,
                                                                  step_sizes=step_sizes,
                                                                  step_weights=step_weights,
                                                                  implementation=dtw_implementation)

        wp = __refine_wp(wp, anchors, wp_list_refine, neighboring_anchors, neighboring_anchor_indices)

        toc2 = time.perf_counter()
        computation_time_it = toc2 - tic2 + toc1 - tic1
        total_computation_time += computation_time_it

        alignment = wp
        feature_rate_old = feature_rate_new
        cost_matrix_size_old = cost_matrix_size_new

        if verbose and cost_matrices_step2 is not None:
            sync_visualize_step2(ax,
                                 cost_matrices_step2,
                                 wp,
                                 wp_step1,
                                 num_rows_step1,
                                 num_cols_step1,
                                 anchors_step1,
                                 neighboring_anchors)
            print('Level {} computation time: {:.2f} seconds'.format(it, computation_time_it))

    if verbose:
        print('Computation time of MrMsDTW: {:.2f} seconds'.format(total_computation_time))

    return alignment


def __refine_wp(wp: np.ndarray,
                anchors: np.ndarray,
                wp_list_refine: list,
                neighboring_anchors: np.ndarray,
                neighboring_anchor_indices: np.ndarray):
    wp_length = wp[:, neighboring_anchor_indices[-1]:].shape[1]
    last_list = wp[:, neighboring_anchor_indices[-1]:] - np.tile(
        wp[:, neighboring_anchor_indices[-1]].reshape(-1, 1), wp_length)
    wp_list_tmp = [wp[:, :neighboring_anchor_indices[0] + 1]] + wp_list_refine + [last_list]
    A_tmp = np.concatenate([anchors[:, 0].reshape(-1, 1), neighboring_anchors, anchors[:, -1].reshape(-1, 1)],
                           axis=1)
    wp_res = build_path_from_warping_paths(warping_paths=wp_list_tmp,
                                           anchors=A_tmp)

    return wp_res
