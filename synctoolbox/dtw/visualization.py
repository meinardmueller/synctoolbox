import matplotlib
import matplotlib.cm
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


def sync_visualize_step1(cost_matrices: list,
                         num_rows: int,
                         num_cols: int,
                         anchors: np.ndarray,
                         wp: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots(1, 1, dpi=72)
    ax = __visualize_cost_matrices(ax, cost_matrices)
    __visualize_constraint_rectangles(anchors[[1, 0], :],
                                      edgecolor='firebrick')

    __visualize_path_in_matrix(ax=ax,
                               wp=wp,
                               axisX=np.arange(0, num_rows),
                               axisY=np.arange(0, num_cols),
                               path_color='firebrick')

    return fig, ax


def sync_visualize_step2(ax: plt.Axes,
                         cost_matrices: list,
                         wp_step2: np.ndarray,
                         wp_step1: np.ndarray,
                         num_rows_step1: int,
                         num_cols_step1: int,
                         anchors_step1: np.ndarray,
                         neighboring_anchors: np.ndarray):

    offset_x = neighboring_anchors[0, 0] - 1
    offset_y = neighboring_anchors[1, 0] - 1
    ax = __visualize_cost_matrices(ax=ax,
                                   cost_matrices=cost_matrices,
                                   offset_x=offset_x,
                                   offset_y=offset_y)

    __visualize_constraint_rectangles(anchors_step1[[1, 0], :],
                                      edgecolor='firebrick')

    __visualize_path_in_matrix(ax=ax,
                               wp=wp_step1,
                               axisX=np.arange(0, num_rows_step1),
                               axisY=np.arange(0, num_cols_step1),
                               path_color='firebrick')

    __visualize_constraint_rectangles(neighboring_anchors[[1, 0], :] - 1,
                                      edgecolor='orangered',
                                      linestyle='--')

    __visualize_path_in_matrix(ax=ax,
                               wp=wp_step2,
                               axisX=np.arange(0, num_rows_step1),
                               axisY=np.arange(0, num_cols_step1),
                               path_color='orangered')

    ax = plt.gca()  # get the current axes
    pcm = None
    for pcm in ax.get_children():
        if isinstance(pcm, matplotlib.cm.ScalarMappable):
            break
    plt.colorbar(pcm, ax=ax)
    plt.tight_layout()
    plt.show()


def __size_dtw_matrices(dtw_matrices: list) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Gives information about the dimensionality of a DTW matrix
    given in form of a list matrix

    Parameters
    ----------
    dtw_matrices: list
        The DTW matrix (cost matrix or accumulated cost matrix) given in form a list.

    Returns
    -------
    axisX_list: list
        A list containing a horizontal axis for each of the sub matrices
        which specifies the horizontal position of the respective submatrix
        in the overall cost matrix.

    axis_y_list: list
        A list containing a vertical axis for each of the
        sub matrices which specifies the vertical position of the
        respective submatrix in the overall cost matrix.

    """
    num_matrices = len(dtw_matrices)
    size_list = [dtw_mat.shape for dtw_mat in dtw_matrices]

    axis_x_list = list()
    axis_y_list = list()

    x_acc = 0
    y_acc = 0

    for i in range(num_matrices):
        curr_size_list = size_list[i]
        axis_x_list.append(np.arange(x_acc, x_acc + curr_size_list[0]))
        axis_y_list.append(np.arange(y_acc, y_acc + curr_size_list[1]))
        x_acc += curr_size_list[0] - 1
        y_acc += curr_size_list[1] - 1

    return axis_x_list, axis_y_list


def __visualize_cost_matrices(ax: plt.Axes,
                              cost_matrices: list = None,
                              offset_x: float = 0.0,
                              offset_y: float = 0.0) -> plt.Axes:
    """Visualizes cost matrices

    Parameters
    ----------
    ax : axes
         The Axes instance to plot on

        The Axes instance to plot on
    cost_matrices : list
        List of DTW cost matrices.

    offset_x : float
        Offset on the x axis.

    offset_y : float
        Offset on the y axis.
    """
    x_ax, y_ax = __size_dtw_matrices(dtw_matrices=cost_matrices)

    for i, cur_cost in enumerate(cost_matrices[::-1]):
        curr_x_ax = x_ax[i] + offset_x
        curr_y_ax = y_ax[i] + offset_y
        cur_cost = cost_matrices[i]
        ax.imshow(cur_cost, cmap='gray_r', aspect='auto', origin='lower',
                  extent=[curr_y_ax[0], curr_y_ax[-1], curr_x_ax[0], curr_x_ax[-1]])

    return ax


def __visualize_path_in_matrix(ax,
                               wp: np.ndarray = None,
                               axisX: np.ndarray = None,
                               axisY: np.ndarray = None,
                               path_color: str = 'r'):
    """Plots a warping path on top of a given matrix. The matrix is
    usually an accumulated cost matrix.

    Parameters
    ----------
    ax : axes
         The Axes instance to plot on

    wp : np.ndarray
        Warping path

    axisX : np.ndarray
        Array of X axis

    axisY : np.ndarray
        Array of Y axis

    path_color : str
        Color of the warping path to be plotted. (default: r)
    """
    assert axisX is not None and isinstance(axisX, np.ndarray), 'axisX must be a numpy array!'
    assert axisY is not None and isinstance(axisY, np.ndarray), 'axisY must be a numpy array!'

    wp = wp.astype(int)

    ax.plot(axisY[wp[1, :]], axisX[wp[0, :]], '-k', linewidth=5)
    ax.plot(axisY[wp[1, :]], axisX[wp[0, :]], color=path_color, linewidth=3)


def __visualize_constraint_rectangles(anchors: np.ndarray,
                                      linestyle: str = '-',
                                      edgecolor: str = 'royalblue',
                                      linewidth: float = 1.0):

    for k in range(anchors.shape[1]-1):
        a1 = anchors[:, k]
        a2 = anchors[:, k + 1]

        # a rectangle is defined by [x y width height]
        x = a1[0]
        y = a1[1]
        w = a2[0] - a1[0] + np.finfo(float).eps
        h = a2[1] - a1[1] + np.finfo(float).eps

        rect = matplotlib.patches.Rectangle((x, y), w, h,
                                            linewidth=linewidth,
                                            edgecolor=edgecolor,
                                            linestyle=linestyle,
                                            facecolor='none')

        plt.gca().add_patch(rect)



