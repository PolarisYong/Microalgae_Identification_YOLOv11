from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ChamberResult:
    source_file: str
    summary_row: int
    chamber: str
    flow: str
    flow_no: int
    chamber_no: int
    umax: float
    k: float
    n0: float
    r2: float
    td: float
    f: float
    eta: float
    lag: float
    log: float
    stable: float

    @property
    def key(self) -> tuple[str, str]:
        return self.source_file, self.chamber

    def output_row(self) -> list[Any]:
        return [
            self.source_file,
            self.flow,
            self.chamber,
            self.umax,
            self.k,
            self.n0,
            self.r2,
            self.td,
            self.f,
            self.eta,
            self.lag,
            self.log,
            self.stable,
        ]


@dataclass(frozen=True)
class GroupStats:
    mean: float
    sample_sd: float
    cv: float
    jb: float
    p_value: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
