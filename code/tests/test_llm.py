import json

import httpx
import pytest

from nl2sql.llm import LLMClient, LLMError
from nl2sql import meter


class Fake:
    def __init__(self, script=None):
        self.n = 0
        self.script = script or []
        self.bodies = []

    def __call__(self, request):
        self.n += 1
        self.bodies.append(json.loads(request.content))
        if self.script:
            code = self.script.pop(0)
            if code != 200:
                return httpx.Response(code, json={"error": {"message": "x", "code": "c"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "SELECT 1"}}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                                                   "prompt_tokens_details": {"cached_tokens": 64}}})


def make(tmp_path, fake):
    kf = tmp_path / "k.env"
    kf.write_text("QWEN_API_KEY=sk-SECRET-TEST\n")
    c = LLMClient(str(tmp_path / "cache"), str(tmp_path / "usage.jsonl"), key_file=str(kf), backoff_base_s=0.01, backoff_cap_s=0.02)
    c._http = httpx.Client(transport=httpx.MockTransport(fake))
    return c


def test_cache_and_usage(tmp_path):
    f = Fake()
    c = make(tmp_path, f)
    msgs = [{"role": "user", "content": "hi"}]
    with meter.Meter() as m:
        a = c.chat("qwen3.7-plus-2026-05-26", msgs, seed=0, tag="gen")
        b = c.chat("qwen3.7-plus-2026-05-26", msgs, seed=0, tag="gen")
    assert f.n == 1 and not a["cached"] and b["cached"]
    assert a["text"] == b["text"] == "SELECT 1"
    assert b["prompt_tokens"] == 100 and b["cached_tokens"] == 64      # hit returns recorded usage
    assert m.calls == 2 and m.prompt_tokens == 200 and m.cached_tokens == 128
    assert f.bodies[0]["enable_thinking"] is False and f.bodies[0]["temperature"] == 0.0
    c.chat("qwen3.7-plus-2026-05-26", msgs, seed=1)            # seed is part of the key
    c.chat("qwen3.7-plus-2026-05-26", msgs, seed=0, temperature=0.7)
    assert f.n == 3
    lines = [json.loads(l) for l in open(tmp_path / "usage.jsonl")]
    assert len(lines) == 4 and lines[1]["cached"] is True and lines[0]["cached_tokens"] == 64
    assert "SECRET" not in open(tmp_path / "usage.jsonl").read()


def test_retry_429_then_ok(tmp_path):
    f = Fake([429, 503, 200])
    c = make(tmp_path, f)
    r = c.chat("qwen3.7-plus-2026-05-26", [{"role": "user", "content": "x"}])
    assert r["text"] == "SELECT 1" and f.n == 3
    assert c.stats["n429"] == 1 and c.stats["n5xx"] == 1 and c.stats["retries"] == 2


def test_gives_up_and_400_not_retried(tmp_path):
    f = Fake([429] * 10)
    c = make(tmp_path, f)
    with pytest.raises(LLMError):
        c.chat("qwen3.7-plus-2026-05-26", [{"role": "user", "content": "y"}])
    assert f.n == 6
    f2 = Fake([400])
    c2 = make(tmp_path / "..", f2) if False else make(tmp_path, f2)
    with pytest.raises(LLMError) as e:
        c2.chat("qwen3.7-plus-2026-05-26", [{"role": "user", "content": "z"}])
    assert f2.n == 1 and e.value.status == 400


def test_forbidden_models(tmp_path):
    c = make(tmp_path, Fake())
    for bad in ("qwen-plus", "qwen-plus-latest", "qwen-max-0919", "qwen-flash", "qwen-turbo", "qwen3-max-preview"):
        with pytest.raises(ValueError):
            c.chat(bad, [{"role": "user", "content": "x"}])
    from nl2sql import config
    for ok in ("qwen3.7-plus-2026-05-26", "qwen3.8-max-0902", "deepseek-v4.1-flash-1017"):
        config.check_model_allowed(ok)


def test_cache_key_stable(tmp_path):
    k1 = LLMClient.cache_key("m", LLMClient._params(0.0, 0, 100, False, 1), [{"role": "user", "content": "a"}])
    k2 = LLMClient.cache_key("m", LLMClient._params(0.0, 0, 100, False, 1), [{"role": "user", "content": "a"}])
    k3 = LLMClient.cache_key("m", LLMClient._params(0.0, 0, 100, True, 1), [{"role": "user", "content": "a"}])
    assert k1 == k2 != k3
