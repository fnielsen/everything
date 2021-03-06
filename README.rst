everything
==========

Liberate your namespace!::

    >>> from everything import *
    >>> sum([Counter(bigrams(list(name))) for name in listdir('.')]).most_common(3)

Rational
--------

Functions, classes and other names needs to be imported in Python before we can use them (except for the builtins).
In interactive data analysis, rather than a "clean" namespace, you might want quick access to a range of names in Python.
Pylab already provides access to many names in Numpy and Matplotlib. This module, called "everything", expands
on Pylab including several other modules relevant for interactive data analysis of data scientists.

The imports are conditional, i.e., a module is not imported if it is not installed.

Examples
--------

Python3 (apropos https://twitter.com/wimlds/status/578678340699045888)::

    >>> from everything import *
    >>> µ = 0.5
    >>> σ = 0.28
    >>> stats.norm(µ, σ)
    <scipy.stats._distn_infrastructure.rv_frozen object at 0x7486090>

(what an ugly name BTW)

Example with NetworkX and NLTK functions::

    >>> from everything import *
    >>> import everything
    >>> g = DiGraph()
    >>> g.add_edges_from(bigrams(word_tokenize(open(everything.__file__.rstrip('c')).read())))
    >>> nx.draw(g, with_labels=True)
    >>> show()

Example with Pandas and Scikit-learn::

    >>> fig, axs = subplots(2, 2)
    >>> for ax, (model, data) in zip(axs.flatten(), itertools.product([PCA(), TSNE()], [load_iris(), load_boston()])):
    ...     DataFrame(model.fit_transform(data.data)).plot(x=0, y=1, kind='scatter', c=data.target / 2., ax=ax)
    >>> show()

Machine learning algorithms with :code:`partial_fit` method::

    >>> import everything
    >>> [name for name in dir(everything) if hasattr(everything.__dict__.get(name), 'partial_fit')]
    ['BernoulliNB', 'GaussianNB', 'MiniBatchKMeans', 'MultinomialNB', 'PassiveAggressiveClassifier',
    'PassiveAggressiveRegressor', 'Perceptron', 'SGDClassifier', 'SGDRegressor']

Machine learning with `auto-sklearn <https://automl.github.io/auto-sklearn/>`_ (requires Python3)

.. code-block:: python
		  
  >>> digits = load_digits()
  >>> X, y = digits.data, digits.target
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
  >>> automl = AutoSklearnClassifier(time_left_for_this_task=30)  # In a hurry
  >>> automl.fit(X_train, y_train)
  >>> y_hat = automl.predict(X_test)
  >>> print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
  Accuracy score 0.986666666667
  
Interactive startup with ipython::

    $ ipython -i -m everything
    
    In [1]: DiGraph()
    Out[1]: <networkx.classes.digraph.DiGraph at 0x7f7f3f92e6d0>

Interactive startup with python::

    $ python -i -c 'from everything import *'
    >>> DiGraph()
    <networkx.classes.digraph.DiGraph object at 0x7f498a54b8d0>


Command-line example
--------------------
:code:`epython` shell script defined as::

    python -i -c 'from everything import *'

:code:`epython3`::

    #!/bin/sh
    python3 -i -c 'from everything import *'

    
Travis et al.
-------------
.. image:: https://travis-ci.org/fnielsen/everything.svg?branch=master
    :target: https://travis-ci.org/fnielsen/everything
