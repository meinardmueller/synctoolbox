[![Python Package using Conda](https://github.com/meinardmueller/synctoolbox/actions/workflows/test_conda.yml/badge.svg)](https://github.com/meinardmueller/synctoolbox/actions/workflows/test_conda.yml)
[![Python package](https://github.com/meinardmueller/synctoolbox/actions/workflows/test_pip.yml/badge.svg)](https://github.com/meinardmueller/synctoolbox/actions/workflows/test_pip.yml)


# Sync Toolbox

This repository contains a Python package called Sync Toolbox, which provides open-source reference implementations for full-fledged music synchronization pipelines and yields state-of-the-art alignment results for a wide range of Western music. 

Using suitable feature representations and cost measures, the toolbox's core technology is based on dynamic time warping (DTW), which brings the feature sequences into temporal correspondence. To account for efficiency, robustness and, accuracy, our toolbox integrates and combines techniques such as [multiscale DTW (MsDTW)](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2006_MuellerMattesKurth_MultiscaleAudioSynchronization_ISMIR.pdf), [memory-restricted MsDTW (MrMsDTW)](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2016_PraetzlichDriedgerMueller_MrMsDTW_ICASSP.pdf), and [high-resolution music synchronization](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2009_EwertMuellerGrosche_HighResAudioSync_ICASSP.pdf). 

If you use the Sync Toolbox in your research, please consider the following references.

## References

Meinard Müller, Henning Mattes, and Frank Kurth.
[An Efficient Multiscale Approach to Audio Synchronization](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2006_MuellerMattesKurth_MultiscaleAudioSynchronization_ISMIR.pdf).
In Proceedings of the International Society for Music Information Retrieval Conference (ISMIR): 192–197, 2006.

Sebastian Ewert, Meinard Müller, and Peter Grosche.
[High Resolution Audio Synchronization Using Chroma Onset Features](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2009_EwertMuellerGrosche_HighResAudioSync_ICASSP.pdf).
In Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP): 1869–1872, 2009.

Thomas Prätzlich, Jonathan Driedger, and Meinard Müller
[Memory-Restricted Multiscale Dynamic Time Warping](https://www.audiolabs-erlangen.de/fau/professor/mueller/publications/2016_PraetzlichDriedgerMueller_MrMsDTW_ICASSP.pdf).
In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP): 569–573, 2016. 

## Installing

If you just want to try our example notebooks, you can run them using Binder directly in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/meinardmueller/synctoolbox/master)

To install the Sync Toolbox locally, you can use the Python package manager pip:

```
pip install synctoolbox
```
We recommend to do this inside a conda or virtual environment. **Note:** On some systems, you may see errors related with ``soundfile`` when calling some functions or executing our example notebooks. ``soundfile`` is a dependency of ``librosa``, which is used by the Sync Toolbox. In case of errors, you may have to install libsndfile using your package manager, e.g., ``sudo apt install libsndfile1``. Alternatively, you may create a conda environment, install ``librosa`` using conda and then install the Sync Toolbox with the pip command from above. See [here](https://github.com/librosa/librosa#hints-for-the-installation) for further information if you are experiencing these issues.


If you want to run the example notebooks locally, you **must** first install the Sync Toolbox to resolve all dependencies. Then, you can clone this repository using

```
git clone https://github.com/meinardmueller/synctoolbox.git
```
install Jupyter using

```
pip install jupyter
```

and then start the notebook server via

```
jupyter notebook
```

Finally, HTML exports of the example notebooks are provided under "[Releases](https://github.com/meinardmueller/synctoolbox/releases)".


## Usage

Fully worked examples for using the sync toolbox are provided in the accompanying Jupyter notebooks. In ``sync_audio_audio_simple.ipynb``, we show how to use the toolbox to synchronize two recordings of the same piece of music using standard chroma features. We also compare runtimes for standard DTW and MrMsDTW. In ``sync_audio_audio_full.ipynb``, we expand this example and demonstrate how to build a full synchronization pipeline that yields state-of-the-art results. Finally, ``sync_audio_score_full.ipynb`` shows a similar pipeline for synchronizing a music recording with the corresponding score.

There is also an API documentation for the Sync Toolbox:

https://meinardmueller.github.io/synctoolbox

## Contributing

We are happy for suggestions and contributions. We would be grateful for either directly contacting us via email (meinard.mueller@audiolabs-erlangen.de) or for creating an issue in our Github repository. Please do not submit a pull request without prior consultation with us.

## Tests

We provide automated tests for each feature and different variants of MrMsDTW. These ensure that the outputs match the ground truth matrices provided in the **tests/data** folder.

To execute the test script, you will need to install extra requirements for testing:

```
pip install 'synctoolbox[tests]'
pytest tests
```

## Licence

The code for this toolbox is published under an MIT licence. This does not apply to the data files. Schubert songs are taken from the [Schubert Winterreise Dataset](https://zenodo.org/record/4122060). The Chopin prelude example files are taken from the [FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html).

## Acknowledgements

The synctoolbox package builds on results, material, and insights that have been obtained in close collaboration with different people. We would like to express our gratitude to former and current students, collaborators, and colleagues who have influenced and supported us in creating this package, including Vlora Arifi-Müller, Michael Clausen, Sebastian Ewert, Christian Fremerey, and Frank Kurth. The main authors of Sync Toolbox are associated with the International Audio Laboratories Erlangen, which are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS. We also thank the German Research Foundation (DFG) for various research grants that allowed us for conducting fundamental research in music processing (in particular, MU 2686/7-2, DFG-MU 2686/14-1).
