"""Testing everything.py module."""


from everything import *


def test_collections():
    assert 'collections' in globals()


def test_collections_27():
    if (2, 7) <= sys.version_info[:2]:
        assert Counter([1, 2, 2]) == {1: 1, 2: 2}


def test_itertools():
    assert 'itertools' in globals()
    assert list(zip_longest((1,), (1, 2))) == [(1, 1), (None, 2)]


def test_json():
    assert 'json' in globals()
    assert json.loads(json.dumps({'a': [2, 3]})) == {'a': [2, 3]}


def test_networkx():
    try:
        find_module('networkx')
    except ImportError: 
        # networkx not available on platform
        pass
    else:
        assert 'nx' in globals()


def test_nltk():
    try:
        find_module('nltk')
    except ImportError:
        # nltk not available on platform
        pass
    else:
        assert 'nltk' in globals()
        assert word_tokenize('Hello world') == ['Hello', 'world']
        assert bigrams([1, 2, 3]) == [(1, 2), (2, 3)]


def test_numpy():
    try:
        find_module('numpy')
    except ImportError:
        pass
    else:
        assert 'np' in globals()
        assert mean([1, 3]) == 2.0


def test_pandas():
    try:
        find_module('pandas')
    except ImportError: 
        pass
    else:
        assert 'pd' in globals()
        assert DataFrame([[1, 2], [3, 4]]).ix[0, 0] == 1
        assert Series([1, 2]).ix[0] == 1


def test_scipy():
    try:
        find_module('scipy')
    except:
        # scipy not available on platform
        pass
    else:
        assert 'scipy' in globals()
        assert 'stats' in globals()
        assert stats.norm(0, 1)
        assert sqrt(-1) == 1j

