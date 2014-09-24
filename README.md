everything
==========

Liberate your namespace!

    >>> from everything import *
    >>> sum([Counter(bigrams(list(name))) for name in listdir('.')]).most_common(3)
