"""Liberate your namespace! Import many modules from Python.

Description
-----------
Import many names (modules, classes, function) from common modules
from Python. All imports from pylab.

Note that this will strongly polute your namespace and it should
probably only be used in interactive programs

Modules, classes and functions:
- base64, b64decode, b64encode, ...
- BeautifulSoup
- bz2
- collections, Counter, defaultdict, ...
- copy (module), deepcopy
- ConfigParser
- glob
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


See also
--------
pylab : Module from matplotlib


Example
-------
>>> import everything
>>> len(dir(everything)) > 140
True

>>> from everything import *
>>> Counter(list(getcwd())).most_common(1)[0][1] > 0
True

"""

__author__ = 'Finn Aarup Nielsen'


# We put this at the top. Do we get any collisions?
try:
    from pylab import *
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

import collections
from collections import Container, Counter, defaultdict, deque, \
    Iterable, Mapping, namedtuple, OrderedDict, Sequence, Sized

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

import itertools
from itertools import count, cycle, repeat, chain, dropwhile, groupby, \
    islice, starmap, tee, takewhile, \
    product, permutations, combinations, \
    combinations_with_replacement
try:
    # Python 3 does not have these functions.
    from itertools import ifilter, ifilterfalse, imap, izip, izip_longest
    zip_longest = izip_longest
except ImportError:
    try:
        from itertools import zip_longest
        ifilter = filter
        izip_longest = zip_longest
        izip = zip
        imap = map
    except ImportError:
        raise

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
    import networkx as nx
    from networkx import DiGraph, Graph, MultiDiGraph, MultiGraph, \
        closeness_centrality, closeness_vitality, \
        connected_component_subgraphs, \
        ego_graph
except ImportError:
    pass

try:
    import nltk
    from nltk import bigrams, pos_tag, sent_tokenize, word_tokenize
except ImportError:
    pass

from operator import attrgetter, itemgetter

from os import chdir, chmod, getcwd, listdir, walk
from os.path import abspath, basename, dirname, exists, expanduser, isdir, \
    isfile, islink, realpath, splitext
from os.path import join as pjoin  # maybe join? No apparent collisions

try:
    import pandas as pd
    from pandas import DataFrame, read_excel, read_csv, \
        Panel, Panel4D, Series
except ImportError:
    pass

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pprint import pprint

# re.compile should not hide __builtins__.compile
# re.split should not hide split from pylab
import re
from re import DOTALL, findall, IGNORECASE, match, MULTILINE, \
    search, sub, subn, UNICODE, VERBOSE

try:
    import requests
except ImportError:
    # Should we import urllib?
    pass

try:
    import scipy
    from scipy.io import loadmat
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
from subprocess import Popen, PIPE, STDOUT

import sys
from sys import stdin, stdout, stderr

import time

from timeit import timeit

try:
    from lxml import etree
except ImportError:
    from xml import etree

import zlib
