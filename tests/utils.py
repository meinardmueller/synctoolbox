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


def dict_allclose(dict1, dict2, atol=1e-5):
    """Checks whether the entries of two dictionaries are close to each other."""
    assert dict1.keys() == dict2.keys()
    for k in dict1:
        print(k, dict1[k].shape, dict2[k].shape)
        assert np.allclose(dict1[k], dict2[k], atol)
