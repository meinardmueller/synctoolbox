import numpy as np
import pickle


def load_dict(filepath):
    """Loads dictionary from a .pickle file.

    Parameters
    ----------
    filepath : str
        Filepath to the .pickle file

    Returns
    -------
    Dictionary
    """
    with open(filepath, 'rb') as byte_stream:
        return pickle.load(byte_stream)


def dict_allclose(dict1, dict2, atol=1e-5, rtol=1e-5):
    """Checks whether the entries of two dictionaries are close to each other."""
    assert dict1.keys() == dict2.keys(), 'The midi indices of two feature arrays are not identical.'
    for k in dict1:
        assert dict1[k].shape == dict2[k].shape, f'Features have different shapes for the midi index {k}.' \
                                                 f'The first feature array f_1[{k}] has {dict1[k].shape} elements;' \
                                                 f'The second feature array f_2[{k}] has {dict2[k].shape} elements:'
        diff = np.abs(dict1[k] - dict2[k])
        idx = diff > atol + rtol * np.abs(dict2[k])
        assert np.allclose(dict1[k].astype(np.float64),
                           dict2[k].astype(np.float64), atol=atol, rtol=rtol), \
            f'The values in the feature arrays don\'t match for the midi index {k},'\
            f'for {idx.size} elements.' \
            f'diff: {diff[idx]}' \
            f'where: {np.where(idx)[0]}'


def filterbank_equal(fb, fb_gt, atol=1e-5):
    """Checks whether the entries of two filterbanks are equal to each other."""
    assert fb.keys() == fb_gt.keys(), 'The MIDI indices of two filterbanks are not identical.'

    for k in fb:
        assert np.allclose(fb[k], fb_gt[k], atol=atol)
