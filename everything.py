"""Liberate your namespace! Import many modules from Python.

Description
-----------
Import many names (modules, classes, function) from common modules
from Python. All imports from pylab.

Note that this will strongly polute your namespace and it should
probably only be used in interactive programs.

Modules, classes and functions:
- base64, b64decode, b64encode, ...
- BeautifulSoup
- bz2
- codecs
- collections, Counter, defaultdict, ...
- copy (module), deepcopy
- ConfigParser
- Decimal from decimal
- glob
- gzip
- json
- pickle
- re, findall, search, sub, subn
- subprocess
- lxml.etree

Conditional import (only if installed):
- DB from db
- networkx
- nltk, sent_tokenize, word_tokenize
- pandas as pd, DataFrame, read_csv, read_excel, Series, ...
- pylab (everything by 'from pylab import *')
- scipy
- sklearn (scikit-learn)


See also
--------
pylab : Module from matplotlib


Example
-------
>>> import everything
>>> len(dir(everything)) > 140
True

>>> from everything import *
>>> if (2, 7) <= sys.version_info[:2]:
...     Counter(list(getcwd())).most_common(1)[0][1] > 0
... else:
...     True
True

"""

__author__ = 'Finn Aarup Nielsen'


# We put this at the top. Do we get any collisions?
try:
    from pylab import *
    from pylab import (arcsin, arcsinh, arctanh, arctan, arctan2, arccosh,
                       arccos)
except ImportError:
    pass

import base64
from base64 import b16decode, b16encode, b32decode, b32encode, \
    b64decode, b64encode, urlsafe_b64decode, urlsafe_b64encode

try:
    from bs4 import BeautifulSoup
except ImportError:
    pass

import bz2

import codecs

import collections
from collections import (Container, Iterable, Mapping, Sequence,
                         Sized, defaultdict, deque, namedtuple)
try:
    from collections import Counter, OrderedDict
except ImportError:
    # Python 2.6
    pass

try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

import copy
from copy import deepcopy

from datetime import date, datetime, timedelta

try:
    from db import DB
except:
    pass

from decimal import Decimal

from fnmatch import fnmatch, fnmatchcase

from functools import partial, wraps

from glob import glob

import gzip

import heapq

from imp import find_module, load_module

import inspect

# 'product' is already imported from pylab
import itertools
from itertools import (chain, combinations, count, cycle, dropwhile,
                       groupby, islice, repeat, starmap,
                       takewhile, tee)
try:
    from itertools import combinations_with_replacement
except ImportError:
    # Python 2.6
    pass
try:
    # Python 3 does not have these functions.
    from itertools import ifilter, ifilterfalse, imap, izip, izip_longest
    zip_longest = izip_longest
except ImportError:
    from itertools import zip_longest
    ifilter = filter
    izip_longest = zip_longest
    izip = zip
    imap = map


import json

# >>> import pylab, math
# >>> set(dir(math)) - set(dir(pylab))
# set(['asin', 'asinh', 'atanh', 'atan', 'atan2', 'factorial',
#      'pow', 'fsum', 'lgamma', 'erf', 'erfc', 'acosh', 'acos'])
try:
    # Numpy's version should work on complex numbers so no need of cmath
    asin = arcsin
    asinh = arcsinh
    atanh = arctanh
    atan = arctan
    atan2 = arctan2
    acosh = arccosh
    acos = arccos
except NameError:
    import math
    from math import *

try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

try:
    import networkx as nx
except ImportError:
    pass
else:
    from networkx import (
        DiGraph, Graph, MultiDiGraph, MultiGraph, closeness_centrality,
        closeness_vitality, connected_component_subgraphs, ego_graph)

try:
    import nltk
except ImportError:
    pass
else:
    from nltk import bigrams, pos_tag, sent_tokenize, word_tokenize

try:
    # Conditional import
    import numpy as np
except ImportError:
    pass

from operator import attrgetter, itemgetter

from os import chdir, chmod, getcwd, listdir, walk
from os.path import (abspath, basename, dirname, exists, expanduser, isdir,
                     isfile, islink, join, realpath, splitext)

try:
    import pandas as pd
except ImportError:
    pass
else:
    from pandas import (DataFrame, read_excel, read_csv,
                        MultiIndex,
                        Panel, Panel4D, Series)

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pprint import pprint

# re.compile should not hide __builtins__.compile
# re.split should not hide split from pylab
import re
from re import (DOTALL, IGNORECASE, MULTILINE, UNICODE, VERBOSE,
                findall, match, search, sub, subn)

try:
    import requests
except ImportError:
    # Should we import urllib?
    pass

try:
    import scipy
except ImportError:
    pass
else:
    import scipy.fftpack
    import scipy.io
    import scipy.io.wavfile
    import scipy.signal
    import scipy.spatial
    from scipy.io import loadmat
    from scipy import stats
    from scipy import signal
    from scipy.signal import iirdesign, iirfilter, periodogram, welch
    from scipy.spatial import ConvexHull

    # Pylab imports Numpy sqrt
    from scipy import sqrt

try:
    import sklearn     # scikit-learn
except ImportError:
    pass
else:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.cross_validation import train_test_split
    from sklearn.datasets import (
        load_boston, load_diabetes, load_digits, load_iris, load_linnerud,
        make_circles, make_moons)
    from sklearn.decomposition import FactorAnalysis, NMF, PCA
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import (
        BayesianRidge, ElasticNet, ElasticNetCV,
        Lasso, LassoLars, LinearRegression, LogisticRegression,
        MultiTaskLasso, OrthogonalMatchingPursuit,
        PassiveAggressiveClassifier, PassiveAggressiveRegressor,
        Perceptron, RANSACRegressor,
        Ridge, RidgeClassifierCV, RidgeCV,
        SGDClassifier, SGDRegressor, TheilSenRegressor)
    from sklearn.manifold import TSNE
    from sklearn.naive_bayes import (BernoulliNB, GaussianNB, MultinomialNB)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import (
        Imputer, PolynomialFeatures, StandardScaler)
    from sklearn.svm import (OneClassSVM, SVC)
    from sklearn.tree import DecisionTreeClassifier

try:
    import sparql
except ImportError:
    pass

try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO

import struct
from struct import pack, unpack

import subprocess
from subprocess import PIPE, Popen, STDOUT

try:
    import sympy
except ImportError:
    pass

import sys
from sys import stderr, stdin, stdout

import time
from time import sleep

from timeit import timeit

try:
    from lxml import etree
except ImportError:
    from xml import etree

import zlib
