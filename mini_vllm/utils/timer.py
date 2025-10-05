import contextlib
import time
from dataclasses import dataclass
from typing import List


# -------------------------------
# Timing utilities
# -------------------------------
@dataclass
class Stat:
    name: str
    ms: float


class Timer:
    def __init__(self):
        self.spans: List[Stat] = []

    @contextlib.contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.spans.append(Stat(name, (t1 - t0) * 1_000))

    def report(self) -> str:
        total = sum(s.ms for s in self.spans)
        lines = [f"Total: {total:.2f} ms"]
        lines += [f" {s.name:<24} {s.ms:8.2f} ms" for s in self.spans]
        return "\n".join(lines)
