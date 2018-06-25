from functools import partial


def identity(x):
    """a -> a"""
    return x


def comp(g, f, *fs):
    """(b -> c) -> (a -> b) -> (a -> c)"""
    if fs: f = comp(f, *fs)
    return lambda x: g(f(x))


def get(k):
    """k -> (k -> v) -> v"""
    return lambda d: d[k]


class Record(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __bool__(self):
        return bool(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, k, d= None):
        return self.__dict__.get(k, d)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__.items())

    def keys(self):
        return self.__dict__.keys()


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
