"""M7 learned controller: numpy forward pass of the trained actor MLP (27->64->64->5, tanh). Weights stored as .npz."""
import numpy as np

from ... import paths


class LearnedController:
    def __init__(self, W, greedy=True, seed=0):
        self.W = W
        self.greedy = greedy
        self.rng = np.random.RandomState(seed)

    @classmethod
    def load(cls, path, greedy=True):
        p = path if str(path).startswith("/") else str(paths.REPO / path)
        z = np.load(p)
        return cls({k: z[k] for k in z.files}, greedy)

    def logits(self, s):
        W = self.W
        h = np.tanh(s @ W["w1"] + W["b1"])
        h = np.tanh(h @ W["w2"] + W["b2"])
        return h @ W["w3"] + W["b3"]

    def act(self, s, mask):
        lg = self.logits(np.asarray(s, dtype=np.float32))
        lg = np.where(np.asarray(mask), lg, -1e9)
        if self.greedy:
            return int(np.argmax(lg))
        p = np.exp(lg - lg.max())
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))
