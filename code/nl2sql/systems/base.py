"""System interface (INTERFACES.md). run() returns one CONTRACT section 4 row WITHOUT `ex`/`exec_ok` (the runner
evaluates). Calls/tokens/latency are filled by the runner from the active Meter when the system does not set them."""
from ..data import Item


class System:
    name = "base"

    def __init__(self, llm, model, cfg):
        self.llm = llm
        self.model = model
        self.cfg = cfg or {}

    def run(self, item: Item, seed: int = 0) -> dict:
        raise NotImplementedError

    def plan_requests(self, item: Item, seed: int = 0):
        """Optional: the first-stage chat requests (dicts accepted by LLMClient.batch_prefill) so that a
        non-interactive stage can be run through the Batch API before the real run (tools/batch_prefill.py)."""
        return None

    def warm(self, items):
        """Optional: build shared indexes/stores before the timed run (called by the runner outside any Meter, so one-time
        setup embedding calls are not attributed to the first question)."""
        return None
