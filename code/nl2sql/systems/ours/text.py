"""Tokenisation, BM25 (incremental inverted index), RRF."""
import math
import re
from collections import Counter, defaultdict

_SPLIT_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[A-Za-z0-9]+")
STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "what", "which", "who",
        "how", "many", "much", "list", "give", "show", "name", "with", "that", "by", "at", "from", "as", "be", "than", "has", "have", "had"}


def tokenize(text, drop_stop=True):
    if not text:
        return []
    text = _SPLIT_CAMEL.sub(" ", str(text)).replace("_", " ")
    toks = [t.lower() for t in _WORD.findall(text)]
    return [t for t in toks if not (drop_stop and t in STOP)]


_QUOTED = re.compile(r"""(['"`])(?:(?!\1).)*\1""")
_NUM = re.compile(r"\b\d[\d,.\-/]*\b")


def mask_question(q):
    """Remove literals (quoted strings, numbers) so BM25 matches question *shape* across databases."""
    q = _QUOTED.sub(" ", q or "")
    return _NUM.sub(" ", q)


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.inv = defaultdict(dict)      # term -> {doc_id: tf}
        self.dl = {}
        self.total = 0

    def add(self, doc_id, tokens):
        if doc_id in self.dl:
            self.remove(doc_id)
        c = Counter(tokens)
        for t, f in c.items():
            self.inv[t][doc_id] = f
        self.dl[doc_id] = len(tokens)
        self.total += len(tokens)

    def remove(self, doc_id):
        if doc_id not in self.dl:
            return
        # linear in vocabulary of the doc is not tracked; scan postings lazily (removals are rare)
        for t in list(self.inv.keys()):
            p = self.inv[t]
            if doc_id in p:
                del p[doc_id]
                if not p:
                    del self.inv[t]
        self.total -= self.dl.pop(doc_id)

    def scores(self, q_tokens, allowed=None):
        N = len(self.dl)
        if N == 0:
            return {}
        avg = self.total / N
        out = defaultdict(float)
        for t in set(q_tokens):
            post = self.inv.get(t)
            if not post:
                continue
            idf = math.log(1 + (N - len(post) + 0.5) / (len(post) + 0.5))
            for d, f in post.items():
                if allowed is not None and d not in allowed:
                    continue
                out[d] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[d] / avg))
        return out


def rrf(rank_lists, k=60, top=None):
    """rank_lists: list of lists of ids (best first). Returns [(id, score)] sorted desc."""
    sc = defaultdict(float)
    for rl in rank_lists:
        for r, i in enumerate(rl):
            sc[i] += 1.0 / (k + r + 1)
    out = sorted(sc.items(), key=lambda x: (-x[1], str(x[0])))
    return out[:top] if top else out
