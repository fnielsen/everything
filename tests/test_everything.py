
from everything import *


def test_collections():
    assert Counter([1, 2, 2]) == {1: 1, 2: 2}


def test_itertools():
    assert list(zip_longest((1,), (1, 2))) == [(1, 1), (None, 2)]


def test_nltk():
    if 'nltk' in sys.modules:
        assert bigrams([1, 2, 3]) == [(1, 2), (2, 3)]
