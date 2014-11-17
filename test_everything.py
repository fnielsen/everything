
from everything import *
import imp

def test_collections():
    assert 'collections' in globals()
    assert Counter([1, 2, 2]) == {1: 1, 2: 2}


def test_itertools():
    assert 'itertools' in globals()
    assert list(zip_longest((1,), (1, 2))) == [(1, 1), (None, 2)]


def test_networkx():
    try:
        imp.find_module('networkx')
        assert 'nx' in globals()
    except:
        # networkx not available on platform
        pass


def test_nltk():
     try:
        imp.find_module('nltk')
        assert 'nltk' in globals()
        assert word_tokenize('Hello world') == ['Hello', 'world']
        assert bigrams([1, 2, 3]) == [(1, 2), (2, 3)]
     except:
         # nltk not available on platform
         pass
