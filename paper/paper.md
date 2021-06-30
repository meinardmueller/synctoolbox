---
title: 'Sync Toolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization'
tags:
  - Python
  - Music information retrieval
  - Music synchronization
authors:
  - name: Meinard Müller
    orcid: 0000-0001-6062-7524
    affiliation: 1
  - name: Yigitcan Özer
    orcid: 0000-0003-2235-8655
    affiliation: 1
  - name: Michael Krause
    orcid: 0000-0001-7194-0719
    affiliation: 1
  - name: Thomas Prätzlich
    affiliation: 1
  - name: Jonathan Driedger
    affiliation: 1
affiliations:
 - name: International Audio Laboratories Erlangen
   index: 1
date: 01 June 2021
bibliography: references.bib
link-citations: yes

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Music can be described and represented in many different ways, including sheet music, symbolic representations, and audio recordings [@Mueller15_FMP_SPRINGER]. For each of these representations, there may exist different versions (e.g., recordings performed by different orchestras and conductors) that correspond to the same musical work. 
Music information retrieval (MIR) aims at developing techniques and tools for organizing, understanding, and searching this information in a robust, efficient, and intelligent manner.
In this context, various alignment and synchronization procedures have been developed with the common goal to automatically link several types of music representations, thus coordinating the multiple information sources related to a given musical work.
In the design and implementation of synchronization algorithms, one has to deal with a delicate tradeoff between efficiency, robustness, and accuracy---requirements leading to various approaches with many design choices.
In this contribution, we introduce a Python package called *Sync Toolbox*, which provides open-source reference implementations for full-fledged music synchronization pipelines and yields state-of-the-art alignment results for a wide range of Western music.
Using suitable feature representations and cost measures, the toolbox's core technology is based on *dynamic time warping* (DTW), which brings the feature sequences into temporal correspondence.
To account for efficiency, robustness and, accuracy, our toolbox integrates and combines techniques such as multiscale DTW (MsDTW) [@MuellerMK06_EfficientMultiscaleApproach_ISMIR; @SalvadorC04_fastDTW], 
memory-restricted MsDTW (MrMsDTW) [@PraetzlichDM16_MsDTW_ICASSP], 
and high-resolution music synchronization [@EwertMG09_HighResAudioSync_ICASSP].
While realizing a complete system with presets that allow users to reproduce research results from the literature, our toolbox also provides well-document functions for all required basic building blocks for feature extraction and alignment.
Furthermore, the toolbox contains example code for visualizing, sonifying, and evaluating synchronization results, thus deepening the understanding of the techniques and data.

# Statement of Need
The task of finding an alignment between two feature sequences has received large research interest in the past, in the context of MIR and beyond. In the music domain, alignment techniques are central for applications such as score following, content-based retrieval, automatic accompaniment, or performance analysis [@Arzt16_MusicTracking_PhD; @Mueller15_FMP_SPRINGER].
Beyond these classical applications, alignment techniques have gained in importance in view of recent data-driven machine learning techniques. In particular, music synchronization has shown its potential for facilitating data annotation, data augmentation, and model evaluation. 
To be more specific, for certain types of music one often has a score-like symbolic representation that explicitly encodes information such as note events, measure positions, lyrics, and other types of metadata. Furthermore, music experts often provide their harmonic, structural, or rhythmic analyses using such symbolic reference representations.
Music synchronization techniques then allow for (semi-)automatically transferring these manually generated annotations from the reference to other symbolic or audio representations. This is beneficial in particular for music, where one has many recorded performances of a given piece. Thus, using music synchronization techniques, one may simplify the annotation process and substantially increase the number of annotated training and test versions.
For example, in [@ZalkowWPAM17_MeasureTransfer_AES], a multi-version approach for transferring measure annotations between music recordings (Wagner operas) is described.
The "Schubert Winterreise Dataset" yields another example where automated techniques were applied to transfer measure, chord, local key, structure, and lyrics annotations [@WeissZAGKVM20_WinterreiseDataset_ACM-JOCCH].
Including nine performances (versions) of Schubert's song cycle, this cross-version dataset was used in [@WeissSM20_LocalKey_TASLP] for training and evaluating data-driven approaches for local key estimation, where the different dataset splits across songs and performances provided new insights into the algorithms' generalization capabilities.

Being a central task, there are a many software packages for sequence alignment of general time series. 
In the audio domain, the Python packages librosa by [@McFeeRLEMBN15_librosa_Python] offers a basic DTW-based pipeline for synchronizing music recordings. 
Since the complexity of alignment techniques such as DTW is proportional to the product of the feature sequences' lengths, runtime and memory requirements become issues when dealing with long feature sequences.
Using a fast online time warping (OLTW) algorithm as described by [@DixonW05_MATCH_ISMIR], the software [^1] (Music Alignment Tool CHest) allows for an efficient alignment of audio files. 
While being efficient, such online approaches are prone to local deviations in the sequences to be aligned. An efficient yet robust alternative is offered by offline procedures based on multiscale strategies such as MsDTW [@MuellerMK06_EfficientMultiscaleApproach_ISMIR; @SalvadorC04_fastDTW]. 
The recent Python package linmdtw [^2] contains an implementation of MsDTW as well as a linear memory DTW variant described in [@TralieD20_DTW_ISMIR].
Another important issue in music synchronization is the temporal accuracy of the alignments, which may be achieved by considering additional local cues such as onset features [@EwertMG09_HighResAudioSync_ICASSP]. Improving the accuracy, however, often goes along with an increase of computational complexity and a decrease of overall robustness.

With our *Sync Toolbox*, we offer a Python package that provides all components to realize a music synchronization pipeline that is robust, efficient, and accurate. 
First, to account for robustness and efficiency it implements the memory-restricted MsDTW approach from [@PraetzlichDM16_MsDTW_ICASSP] as its algorithmic core component.
Second, to account for accuracy, it integrates the high-resolution strategy from [@EwertMG09_HighResAudioSync_ICASSP] on the finest MsDTW layer.
Third, the toolbox contains all feature extractions methods (including chroma and onset features) needed to reproduce the results from the research literature. 
Fourth, we also provide functions required for quantitative and qualitative evaluations (including visualization and sonification methods).
Even though having an overlap to the previously mentioned software (e.g., librosa and linmdtw), the Sync Toolbox provides for the first time an open-source Python package for offline music synchronization that produces state-of-the-art alignment results regarding efficiency and accuracy.
<!---
For example, given pre-computed feature representations, the overall alignment of 20-minute recordings at a feature resolution of $50$~Hz (corresponding to 20 milliseconds) requires roughly ??? MB of additional memory (besides the memory required to store the features) and takes ??? seconds on a standard PC.
-->
Thus, with the publicly available and well-documented Sync Toolbox, we hope to fill a gap between theory and practice for an important MIR task, while providing a useful pre-processing, annotation, and evaluation tool for data-driven machine learning. 


[^1]: <http://www.eecs.qmul.ac.uk/~simond/match/>
[^2]: <https://github.com/ctralie/linmdtw>


# Design Choices
When we designed the Sync Toolbox, we had different objectives in mind. First, we tried to keep a close connection to the research articles [@EwertMG09_HighResAudioSync_ICASSP] and [@PraetzlichDM16_MsDTW_ICASSP]. 
Second,  we reimplemented and included all required components (e.g., feature extractors, DTW), even though such basic functionality is also covered by other packages such as librosa and linmdtw. This way, along with a specification of meaningful variable preset, the Sync Toolbox provids reference implementations for exactly reproducing previously published research results and experiments.
Third, we followed many of the design principles suggested by librosa [@McFeeRLEMBN15_librosa_Python], which allows users to easily combine the different Python packages. 
The code of the Sync Toolbox along with an API documentation is hosted in a publicly available GitHub repository. [^3]
Finally, we included the synctoolbox package into the Python package index PyPi, which makes it possible to install synctoolbox with the standard Python package manager pip. [^4]

[^3]: <https://github.com/meinardmueller/synctoolbox>
[^4]: <https://pypi.org/project/synctoolbox>

# Acknowledgements
The synctoolbox package builds on results, material, and insights that have been obtained in close collaboration with different people. We would like to express our gratitude to former and current students, collaborators, and colleagues who have influenced and supported us in creating this package, including Vlora Arifi-Müller, Michael Clausen, Sebastian Ewert, Christian Fremerey, and Frank Kurth. We also thank the German Research Foundation (DFG) for various research grants that allowed us for conducting fundamental research in music processing (in particular, MU 2686/7-2, DFG-MU 2686/14-1). The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

# References
