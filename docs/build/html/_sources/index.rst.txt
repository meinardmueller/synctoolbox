.. SyncToolbox documentation master file, created by
   sphinx-quickstart on Tue May  4 08:49:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization
========================================================================================
**Sync Toolbox** is a Python package, which comprises all components of a music synchronization
pipeline that is robust, efficient, and accurate.

The toolbox's core technology is based on dynamic time warping (DTW). Using suitable feature representations
and cost measures, DTW brings the feature sequences into temporal correspondence. To account for efficiency,
robustness, and accuracy, **Sync Toolbox** uses a combination of
`multiscale DTW (MsDTW) <https://www.audiolabs-erlangen.de/content/05-fau/professor/00-mueller/03-publications/2006_MuellerMattesKurth_MultiscaleAudioSynchronization_ISMIR.pdf>`__,
`memory-restricted MsDTW (MrMsDTW) <https://www.audiolabs-erlangen.de/content/05-fau/professor/00-mueller/03-publications/2016_PraetzlichDriedgerMueller_MrMsDTW_ICASSP.pdf>`__,
and `high-resolution music synchronization <https://www.audiolabs-erlangen.de/content/05-fau/professor/00-mueller/03-publications/2009_EwertMuellerGrosche_HighResAudioSync_ICASSP.pdf>`__.

Despite a slight overlap with the well-known software packages in the MIR field (e.g., librosa and linmdtw),
our **Sync Toolbox** is the first to provide an open-source Python package for offline music synchronization
that produces state-of-the-art alignment results regarding efficiency and accuracy.



Sync Toolbox API Documentation
------------------------------

The source code for the package Sync Toolbox is hosted at GitHub:

https://github.com/meinardmueller/synctoolbox

In particular, please note the provided Readme and the example notebooks.

If you use SyncToolbox in a scholarly work, please consider citing the Sync Toolbox article. [#]_


.. [#] Müller et al., (2021). Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization. Journal of Open Source Software, 6(64), 3434, https://doi.org/10.21105/joss.03434

.. toctree::
    :caption: API Documentation
    :maxdepth: 2
    :hidden:

    dtw
    feature/index


.. toctree::
    :caption: Reference
    :maxdepth: 1
    :hidden:

    genindex
    py-modindex
