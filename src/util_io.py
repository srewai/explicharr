from collections import Counter, deque


def load(filename):
    with open(filename) as file:
        for line in file:
            yield line


def save(filename, lines):
    with open(filename, 'w') as file:
        for line in lines:
            print(line, file= file)


def chartab(corpus, top= 256, special= "\xa0\n "):
    """returns the `top` most frequent characters in `corpus`, and ensures
    that the `special` characters are included with the highest ranks.

    corpus : seq (line : str)

    """

    char2freq = Counter(char for line in corpus for char in line)
    for char in special: del char2freq[char]
    return special + "".join([k for k, _ in sorted(
        char2freq.items()
        , key= lambda kv: (-kv[1], kv[0])
    )[:top-len(special)]])


def encode(index, sent, end= "\n", start= " "):
    """-> list int

    encodes `sent : seq str` according to `index : PointedIndex`.

    ensures that it starts with `start` and ends with `end`.

    """
    sent = deque(sent)
    if sent and not sent[0] == start: sent.appendleft(start)
    if sent and not sent[-1] == end: sent.append(end)
    return list(map(index, sent))


def decode(index, idxs, end= "\n", sep= ""):
    """-> str

    decodes `idxs : seq int` according to `index : PointedIndex`.

    stops at `end` and joins the results with `sep`.

    """
    end = index(end)
    tgt = []
    for idx in idxs:
        if idx == end: break
        tgt.append(index[idx])
    return sep.join(tgt)
