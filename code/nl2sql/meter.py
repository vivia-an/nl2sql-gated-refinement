"""Per-question accounting via contextvars. The runner wraps System.run in `with Meter() as m:`;
LLMClient/db.execute add to the active meter, so baselines get calls/tokens without extra code.
Stage-level breakdown (for cost analysis) is keyed by the `tag` argument of LLMClient.chat."""
import contextvars
import threading

_CUR = contextvars.ContextVar("nl2sql_meter", default=None)


class Meter:
    def __init__(self):
        self.calls = 0
        self.cached_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0      # implicit-context-cache tokens reported by the API (subset of prompt_tokens)
        self.latency_s = 0.0        # sum of LLM call latencies (cache hit = original latency)
        self.exec_s = 0.0           # sum of SQL execution times
        self.embed_calls = 0
        self.rerank_calls = 0
        self.stages = {}            # tag -> {calls, prompt, completion, cached, latency}
        self._lock = threading.Lock()
        self._tok = None
        self._parent = None

    def __enter__(self):
        self._parent = _CUR.get()          # nested meters forward to the enclosing one
        self._tok = _CUR.set(self)
        return self

    def __exit__(self, *a):
        _CUR.reset(self._tok)

    def add_llm(self, tag, pt, ct, cached_tokens, latency, was_cached):
        if self._parent:
            self._parent.add_llm(tag, pt, ct, cached_tokens, latency, was_cached)
        with self._lock:
            self.calls += 1
            self.cached_calls += int(was_cached)
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.cached_tokens += cached_tokens
            self.latency_s += latency
            s = self.stages.setdefault(tag or "untagged", {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "latency_s": 0.0})
            s["calls"] += 1
            s["prompt"] += pt
            s["completion"] += ct
            s["cached"] += cached_tokens
            s["latency_s"] += latency

    def add_exec(self, secs):
        if self._parent:
            self._parent.add_exec(secs)
        with self._lock:
            self.exec_s += secs

    def add_aux(self, kind):
        if self._parent:
            self._parent.add_aux(kind)
        with self._lock:
            if kind == "embed":
                self.embed_calls += 1
            else:
                self.rerank_calls += 1

    def summary(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cached_tokens": self.cached_tokens, "latency_s": round(self.latency_s + self.exec_s, 3),
                "llm_latency_s": round(self.latency_s, 3), "exec_s": round(self.exec_s, 3),
                "embed_calls": self.embed_calls, "rerank_calls": self.rerank_calls,
                "stages": {k: dict(v) for k, v in self.stages.items()}}


def current():
    return _CUR.get()
