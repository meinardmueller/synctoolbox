from setuptools import setup, find_packages


with open('README.md', 'r') as fdesc:
    long_description = fdesc.read()

setup(
    name='synctoolbox',
    version='1.0.2',
    description='Python Package for Efficient, Robust, and Accurate Music Synchronization (SyncToolbox)',
    author='Meinard Müller, Yigitcan Özer, Michael Krause, Thomas Prätzlich and Jonathan Driedger',
    author_email='meinard.mueller@audiolabs-erlangen.de',
    url='https://github.com/meinardmueller/synctoolbox',
    download_url='https://github.com/meinardmueller/synctoolbox',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3",
    ],
    keywords='audio music sound synchronization dtw mrmsdtw',
    license='MIT',
    install_requires=['ipython >= 7.8.0, < 8.0.0',
                      'librosa >= 0.8.0, < 1.0.0',
                      'matplotlib >= 3.1.0, < 4.0.0',
                      'music21 >= 5.7.0, < 6.0.0',
                      'numba >= 0.51.0, < 1.0.0',
                      'numpy >= 1.17.0, < 2.0.0',
                      'pandas >= 1.0.0, < 2.0.0',
                      'pretty_midi >= 0.2.0, < 1.0.0',
                      'pysoundfile >= 0.9.0, < 1.0.0',
                      'scipy >= 1.7.0, < 2.0.0',
                      'libfmp >= 1.2.0, < 2.0.0'],
    python_requires='>=3.7, <4.0',
    extras_require={
        'tests': ['pytest == 6.2.*'],
        'docs': ['sphinx == 4.0.*',
                 'sphinx_rtd_theme == 0.5.*']
    }
)
