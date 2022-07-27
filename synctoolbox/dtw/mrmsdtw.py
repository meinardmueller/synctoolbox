from numba import jit
import numpy as np
import time
from typing import List, Tuple, Optional

from synctoolbox.dtw.anchor import derive_anchors_from_projected_alignment, derive_neighboring_anchors, \
    project_alignment_on_a_new_feature_rate
from synctoolbox.dtw.utils import build_path_from_warping_paths, compute_cost_matrices_between_anchors, \
    compute_warping_paths_from_cost_matrices, find_anchor_indices_in_warping_path
from synctoolbox.dtw.visualization import sync_visualize_step1, sync_visualize_step2
from synctoolbox.feature.utils import smooth_downsample_feature, normalize_feature


def sync_via_mrmsdtw_with_anchors(f_chroma1: np.ndarray,
                                  f_chroma2: np.ndarray,
                                  f_onset1: np.ndarray = None,
                                  f_onset2: np.ndarray = None,
                                  input_feature_rate: int = 50,
                                  step_sizes: np.ndarray = np.array([[1, 0], [0, 1], [1, 1]], np.int32),
                                  step_weights: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64),
                                  threshold_rec: int = 10000,
                                  win_len_smooth: np.ndarray = np.array([201, 101, 21, 1]),
                                  downsamp_smooth: np.ndarray = np.array([50, 25, 5, 1]),
                                  verbose: bool = False,
                                  dtw_implementation: str = 'synctoolbox',
                                  normalize_chroma: bool = True,
                                  chroma_norm_ord: int = 2,
                                  chroma_norm_threshold: float = 0.001,
                                  visualization_title: str = "MrMsDTW result",
                                  constant_intervals: List[Tuple[Tuple, Tuple, bool]] = None) -> np.ndarray:
    """Compute memory-restricted multi-scale DTW (MrMsDTW) using chroma and (optionally) onset features.
        MrMsDTW is performed on multiple levels that get progressively finer, with rectangular constraint
        regions defined by the alignment found on the previous, coarser level.
        If onset features are provided, these are used on the finest level in addition to chroma
        to provide higher synchronization accuracy.

        Parameters
        ----------
        f_chroma1 : np.ndarray [shape=(12, N)]
            Chroma feature matrix of the first sequence

        f_chroma2 : np.ndarray [shape=(12, M)]
            Chroma feature matrix of the second sequence

        f_onset1 : np.ndarray [shape=(L, N)]
            Onset feature matrix of the first sequence (optional, default: None)

        f_onset2 : np.ndarray [shape=(L, M)]
            Onset feature matrix of the second sequence (optional, default: None)

        input_feature_rate: int
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

        visualization_title : str
            Title for the visualization plots. Only relevant if 'verbose' is True
            (default: "MrMsDTW result")

        constant_intervals: List[Tuple[Tuple, Tuple, bool]]
            Constant intervals in the following format

            e.g., [((0,  x1), (0, y1), False),
                   ((x1, x2), (y1, y2), True),
                   ((x2, -1), (y2, -1), False)]

            * The boolean value determines whether MrMsDTW should be applied in the interval.
                - If the interval is only silence, it usually does not make sense to synchronize that interval.
            * The starting anchor point of an interval must be the ending anchor point of the previous interval.
            * The last interval can end with -1, meaning that the interval ends at the end of the feature sequence.

        Returns
        -------
        wp : np.ndarray [shape=(2, T)]
            Resulting warping path
    """
    if constant_intervals is None:
        wp = sync_via_mrmsdtw(f_chroma1=f_chroma1,
                              f_chroma2=f_chroma2,
                              f_onset1=f_onset1,
                              f_onset2=f_onset2,
                              input_feature_rate=input_feature_rate,
                              step_sizes=step_sizes,
                              step_weights=step_weights,
                              threshold_rec=threshold_rec,
                              win_len_smooth=win_len_smooth,
                              downsamp_smooth=downsamp_smooth,
                              verbose=verbose,
                              dtw_implementation=dtw_implementation,
                              normalize_chroma=normalize_chroma,
                              chroma_norm_ord=chroma_norm_ord,
                              chroma_norm_threshold=chroma_norm_threshold,
                              visualization_title=visualization_title)
    else:
        # constant_intervals = [((0,  x1), (0, y1), False),
        #                       ((x1, x2), (y1, y2), True),
        #                       ((x2, -1), (y2, -1), False)]
        wp = None
        __check_constant_intervals(constant_intervals, f_chroma1.shape[1], f_chroma2.shape[1], input_feature_rate)

        if verbose:
            print('Predefined anchor points are detected!')

        for constant_interval in constant_intervals:
            (f1_start, f1_end), (f2_start, f2_end), run_algorithm = constant_interval

            # Split the features
            f_chroma1_split, f_onset1_split, f_chroma2_split, f_onset2_split = __split_features(f_chroma1,
                                                                                                f_onset1,
                                                                                                f_chroma2,
                                                                                                f_onset2,
                                                                                                constant_interval,
                                                                                                input_feature_rate)

            if not run_algorithm:
                # Generate a diagonal warping path, if the algorithm is not supposed to executed.
                # A typical scenario is the silence breaks which are enclosed by two anchor points.
                if verbose:
                    print('A diagonal warping path is generated for the interval \n\t Feature sequence 1: %.2f - %.2f'
                          '\n\t Feature sequence 2: %.2f - %.2f\n' % (f1_start, f1_end, f2_start, f2_end))
                wp_cur = __diagonal_warping_path(f_chroma1_split, f_chroma2_split)
                wp_cur *= input_feature_rate / 1000

            else:
                if verbose:
                    print('MrMsDTW is applied for the interval \n\t Feature sequence 1: %.2f - %.2f'
                          '\n\t Feature sequence 2: %.2f - %.2f\n' % (f1_start, f1_end, f2_start, f2_end))

                wp_cur = sync_via_mrmsdtw(f_chroma1=f_chroma1_split,
                                          f_chroma2=f_chroma2_split,
                                          f_onset1=f_onset1_split,
                                          f_onset2=f_onset2_split,
                                          input_feature_rate=input_feature_rate,
                                          step_sizes=step_sizes,
                                          step_weights=step_weights,
                                          threshold_rec=threshold_rec,
                                          win_len_smooth=win_len_smooth,
                                          downsamp_smooth=downsamp_smooth,
                                          verbose=verbose,
                                          dtw_implementation=dtw_implementation,
                                          normalize_chroma=normalize_chroma,
                                          chroma_norm_ord=chroma_norm_ord,
                                          chroma_norm_threshold=chroma_norm_threshold)

            if wp is None:
                wp = np.array(wp_cur, copy=True)

            # Concatenate warping paths
            else:
                wp = np.concatenate([wp, wp_cur[:, 1:] + wp[:, -1].reshape(2, 1) + 1], axis=1)
    return wp


def sync_via_mrmsdtw(f_chroma1: np.ndarray,
                     f_chroma2: np.ndarray,
                     f_onset1: np.ndarray = None,
                     f_onset2: np.ndarray = None,
                     input_feature_rate: int = 50,
                     step_sizes: np.ndarray = np.array([[1, 0], [0, 1], [1, 1]], np.int32),
                     step_weights: np.ndarray = np.array([1.0, 1.0, 1.0], np.float64),
                     threshold_rec: int = 10000,
                     win_len_smooth: np.ndarray = np.array([201, 101, 21, 1]),
                     downsamp_smooth: np.ndarray = np.array([50, 25, 5, 1]),
                     verbose: bool = False,
                     dtw_implementation: str = 'synctoolbox',
                     normalize_chroma: bool = True,
                     chroma_norm_ord: int = 2,
                     chroma_norm_threshold: float = 0.001,
                     visualization_title: str = "MrMsDTW result",
                     alpha=0.5) -> np.ndarray:
    """Compute memory-restricted multi-scale DTW (MrMsDTW) using chroma and (optionally) onset features.
        MrMsDTW is performed on multiple levels that get progressively finer, with rectangular constraint
        regions defined by the alignment found on the previous, coarser level.
        If onset features are provided, these are used on the finest level in addition to chroma
        to provide higher synchronization accuracy.

        Parameters
        ----------
        f_chroma1 : np.ndarray [shape=(12, N)]
            Chroma feature matrix of the first sequence

        f_chroma2 : np.ndarray [shape=(12, M)]
            Chroma feature matrix of the second sequence

        f_onset1 : np.ndarray [shape=(L, N)]
            Onset feature matrix of the first sequence (optional, default: None)

        f_onset2 : np.ndarray [shape=(L, M)]
            Onset feature matrix of the second sequence (optional, default: None)

        input_feature_rate: int
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

        visualization_title : str
            Title for the visualization plots. Only relevant if 'verbose' is True
            (default: "MrMsDTW result")
    """
    # If onset features are given as input, high resolution MrMsDTW is activated.
    high_res = False
    if f_onset1 is not None and f_onset2 is not None:
        high_res = True

    if high_res and (f_chroma1.shape[1] != f_onset1.shape[1] or f_chroma2.shape[1] != f_onset2.shape[1]):
        raise ValueError('Chroma and onset features must be of the same length.')

    if downsamp_smooth[-1] != 1 or win_len_smooth[-1] != 1:
        raise ValueError('The downsampling factor of the last iteration must be equal to 1, i.e.'
                         'at the last iteration, it is computed at the input feature rate!')

    num_iterations = win_len_smooth.shape[0]
    cost_matrix_size_old = tuple()
    feature_rate_old = input_feature_rate / downsamp_smooth[0]
    alignment = None
    total_computation_time = 0.0

    # If the area is less than the threshold_rec, don't apply the multiscale DTW.
    it = (num_iterations - 1) if __compute_area(f_chroma1, f_chroma2) < threshold_rec else 0

    while it < num_iterations:
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
                                                                        f_onset1=f_onset1,
                                                                        f_onset2=f_onset2,
                                                                        anchors=anchors,
                                                                        alpha=alpha)

        else:
            cost_matrices_step1 = compute_cost_matrices_between_anchors(f_chroma1=f_chroma1_cur,
                                                                        f_chroma2=f_chroma2_cur,
                                                                        anchors=anchors,
                                                                        alpha=alpha)

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
                                                                        f_onset1=f_onset1,
                                                                        f_onset2=f_onset2,
                                                                        anchors=neighboring_anchors,
                                                                        alpha=alpha)

        else:
            cost_matrices_step2 = compute_cost_matrices_between_anchors(f_chroma1=f_chroma1_cur,
                                                                        f_chroma2=f_chroma2_cur,
                                                                        anchors=neighboring_anchors,
                                                                        alpha=alpha)

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
                                 neighboring_anchors,
                                 plot_title=f"{visualization_title} - Level {it + 1}")
            print('Level {} computation time: {:.2f} seconds'.format(it, computation_time_it))

        it += 1

    if verbose:
        print('Computation time of MrMsDTW: {:.2f} seconds'.format(total_computation_time))

    return alignment


def __diagonal_warping_path(f1: np.ndarray,
                            f2: np.ndarray) -> np.ndarray:
    """Generates a diagonal warping path given two feature sequences.

    Parameters
    ----------
    f1: np.ndarray [shape=(_, N)]
        First feature sequence

    f2: np.ndarray [shape=(_, M)]
        Second feature sequence

    Returns
    -------
    np.ndarray: Diagonal warping path [shape=(2, T)]
    """
    max_size = np.maximum(f1.size, f2.size)
    if max_size == f1.size:
        return np.array([np.linspace(0, max_size-1, max_size), np.round(np.linspace(0, f2.size-1, max_size))])
    else:
        return np.array([np.round(np.linspace(0, f1.size, max_size)), np.linspace(0, max_size-1, max_size)])


@jit(nopython=True)
def __compute_area(f1, f2):
    """Computes the area of the cost matrix given two feature sequences

    Parameters
    ----------
    f1: np.ndarray
        First feature sequence

    f2: np.ndarray
        Second feature sequence

    Returns
    -------
    int: Area of the cost matrix
    """
    return f1.shape[1] * f2.shape[1]


def __split_features(f_chroma1: np.ndarray,
                     f_onset1: np.ndarray,
                     f_chroma2: np.ndarray,
                     f_onset2: np.ndarray,
                     constant_interval: Tuple[Tuple, Tuple, bool],
                     feature_rate: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    (f1_start, f1_end), (f2_start, f2_end), run_algorithm = constant_interval

    # Split the features
    if f1_end != -1:
        f_chroma1_split = f_chroma1[:, int(f1_start * feature_rate):int(f1_end * feature_rate)]
        if f_onset1 is not None:
            f_onset1_split = f_onset1[:, int(f1_start * feature_rate):int(f1_end * feature_rate)]
    else:
        f_chroma1_split = f_chroma1[:, int(f1_start * feature_rate):]
        if f_onset1 is not None:
            f_onset1_split = f_onset1[:, int(f1_start * feature_rate):]

    if f2_end != -1:
        f_chroma2_split = f_chroma2[:, int(f2_start * feature_rate):int(f2_end * feature_rate)]
        if f_onset2 is not None:
            f_onset2_split = f_onset2[:, int(f2_start * feature_rate):int(f2_end * feature_rate)]

    else:
        f_chroma2_split = f_chroma2[:, int(f2_start * feature_rate):]
        if f_onset2 is not None:
            f_onset2_split = f_onset2[:, int(f2_start * feature_rate):]

    if f_onset1 is None:
        f_onset1_split = None

    if f_onset2 is None:
        f_onset2_split = None

    return f_chroma1_split, f_onset1_split, f_chroma2_split, f_onset2_split


def __refine_wp(wp: np.ndarray,
                anchors: np.ndarray,
                wp_list_refine: List,
                neighboring_anchors: np.ndarray,
                neighboring_anchor_indices: np.ndarray) -> np.ndarray:
    wp_length = wp[:, neighboring_anchor_indices[-1]:].shape[1]
    last_list = wp[:, neighboring_anchor_indices[-1]:] - np.tile(
        wp[:, neighboring_anchor_indices[-1]].reshape(-1, 1), wp_length)
    wp_list_tmp = [wp[:, :neighboring_anchor_indices[0] + 1]] + wp_list_refine + [last_list]
    A_tmp = np.concatenate([anchors[:, 0].reshape(-1, 1), neighboring_anchors, anchors[:, -1].reshape(-1, 1)],
                           axis=1)
    wp_res = build_path_from_warping_paths(warping_paths=wp_list_tmp,
                                           anchors=A_tmp)

    return wp_res


def __check_constant_intervals(constant_intervals: List,
                               f_len1: int,
                               f_len2: int,
                               feature_rate: int):
    """Ensures that the predefined anchors satisfy the conditions

    Parameters
    ----------
    constant_intervals: List[List[Tuple, Tuple, bool]]
        Constant intervals in the following format
        e.g., [((0,  x1), (0, y1), False),
               ((x1, x2), (y1, y2), True),
               ((x2, -1), (y2, -1), False)]

    f_len1: int
        Length of the first feature sequence

    f_len2: int
        Length of the second feature sequence

    feature_rate: int
        Input feature rate of the features
    """

    prev_end_1 = None
    prev_end_2 = None

    for constant_interval in constant_intervals:
        (f1_start, f1_end), (f2_start, f2_end), run_algorithm = constant_interval
        if f1_start < 0 or f2_start < 0:
            raise ValueError('Starting point must be a positive number!')

        if (not f1_end == -1 and f1_end <= f1_start) or (not f2_end == -1 and f2_end <= f2_start):
            raise ValueError('Ending point of an interval must be greater than the starting point of an interval!')

        if f1_start > f_len1 / feature_rate or f1_end > f_len1 / feature_rate or \
                f2_start > f_len2 / feature_rate or f2_end > f_len2 / feature_rate:
            raise ValueError('Anchor points must be given in seconds between 0 and the length of the corresponding '
                             'audio.')

        if prev_end_1 is not None and prev_end_2 is not None:
            if f1_start != prev_end_1 and f2_start != prev_end_2:
                raise ValueError('The starting point of an interval must be the equal '
                                 'to the ending point of the previous interval!')

        prev_end_1 = f1_end
        prev_end_2 = f2_end
