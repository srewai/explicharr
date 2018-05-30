from collections import Counter


def load(filename, proc= str.split):
    """-> list line;  proc : str -> line"""
    with open(filename) as file:
        return list(map(proc, file))


def chartab(corpus, top= 256, special= "\xa0\n "):
    """returns the `top` most frequent characters in `corpus`, and ensures
    that the `special` characters are included with the highest ranks.

    corpus : seq (sent : seq (word : str))

    """
    char2freq = Counter(char for sent in corpus for word in sent for char in word)
    for char in special: del char2freq[char]
    return special + "".join([k for k, _ in sorted(
        char2freq.items()
        , key= lambda kv: (-kv[1], kv[0])
    )[:top-len(special)]])


class PointedIndex:
    """takes a vector of unique elements `vec` and a base index `nil`
    within its range, returns a pointed index `idx`, such that:

    `idx[i]` returns the element at index `i`;

    `idx(x)` returns the index of element `x`;

    bijective for all indices and elements within `vec`;

    otherwise `idx(x) => nil` and `idx[i] => vec[nil]`.

    """

    def __init__(self, vec, nil= 0):
        self._nil = nil
        self._i2x = vec
        self._x2i = {x: i for i, x in enumerate(vec)}

    def __repr__(self):
        return "{}(vec= {}, nil= {})".format(
            type(self).__name__
            , repr(self._i2x)
            , repr(self._nil))

    def __getitem__(self, i):
        try:
            return self._i2x[i]
        except IndexError:
            return self._i2x[self._nil]

    def __call__(self, x):
        try:
            return self._x2i[x]
        except KeyError:
            return self._nil

    def __len__(self):
        return len(self._i2x)

    @property
    def vec(self):
        return self._i2x

    @property
    def nil(self):
        return self._nil


if "__main__" == __name__:
    path = "../data/"
    normal = load(path + "train.nen")
    normal_idx = PointedIndex(chartab(normal))
