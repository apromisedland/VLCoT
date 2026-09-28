import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable


OPERATIONS = ("encode", "initial", "rolling", "verify", "action", "repair")


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    limit: float
    unit_costs: dict[str, float]
    synchronize: Callable[[], None] = lambda: None
    clock: Callable[[], float] = time.perf_counter
    spent: float = 0.0
    records: dict = field(default_factory=dict)
    _active: bool = False

    def __post_init__(self):
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("A finite positive task budget is required")
        if set(self.unit_costs) != set(OPERATIONS):
            raise ValueError(f"Cost profile must define exactly {OPERATIONS}")
        if any(not math.isfinite(value) or value <= 0 for value in self.unit_costs.values()):
            raise ValueError("Every model operation must have a finite positive cost")
        self.records = {name: {"calls": 0, "units": 0.0, "seconds": 0.0} for name in OPERATIONS}

    def permits(self, operation: str, count: int = 1) -> bool:
        return count > 0 and self.spent + self.unit_costs[operation] * count <= self.limit + 1e-9

    @contextmanager
    def charge(self, operation: str, count: int = 1):
        if self._active:
            raise RuntimeError("Nested accounting would double-count model time")
        if not self.permits(operation, count):
            raise BudgetExhausted(f"Insufficient budget for {operation} ({count} units)")
        self.synchronize()
        start = self.clock()
        cost = self.unit_costs[operation] * count
        self.spent += cost
        record = self.records[operation]
        record["calls"] += 1
        record["units"] += cost
        self._active = True
        try:
            yield
        finally:
            try:
                self.synchronize()
            finally:
                record["seconds"] += self.clock() - start
                self._active = False

    @property
    def seconds(self):
        return sum(record["seconds"] for record in self.records.values())
