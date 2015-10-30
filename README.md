everything
==========

Liberate your namespace!

    >>> from everything import *
    >>> sum([Counter(bigrams(list(name))) for name in listdir('.')]).most_common(3)

Python3 (apropos https://twitter.com/wimlds/status/578678340699045888):

    >>> from everything import *
    >>> µ = 0.5
    >>> σ = 0.28
    >>> stats.norm(µ, σ)
    <scipy.stats._distn_infrastructure.rv_frozen object at 0x7486090>

(what an ugly name BTW)

Example with NetworkX and NLTK functions:

    >>> from everything import *
    >>> import everything
    >>> g = DiGraph()
    >>> g.add_edges_from(bigrams(word_tokenize(open(everything.__file__.rstrip('c')).read())))
    >>> nx.draw(g, with_labels=True)
    >>> show()