"""ActionLogger extracted unchanged from q3-v1.99."""
import os
from typing import Optional

class ActionLogger:
    """结构化动作日志：每条动作一行，便于赛后统计与复盘。"""

    def __init__(self, path: Optional[str] = None, verbose: bool = False):
        self.path = path
        self.verbose = verbose
        self._header_written = False
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "seq\ttime_s\tphase\taction\tx\ty\tchannel\tresult\tsvd_deg\t"
                    "channel_status\tfeasible_radius\tnote\n"
                )
            self._header_written = True

    def log(self, seq: int, virtual_time: float, phase: str, action: str,
            x: float, y: float, channel: int, result: str = "-",
            svd_deg: Optional[float] = None, channel_status: str = "-",
            feasible_radius: Optional[float] = None, note: str = "") -> None:
        svd_text = "-" if svd_deg is None else f"{svd_deg:.3f}"
        radius_text = "-" if feasible_radius is None else f"{feasible_radius:.3f}"
        line = (
            f"{seq}\t{virtual_time:.3f}\t{phase}\t{action}\t{x:.3f}\t{y:.3f}\t"
            f"{channel}\t{result}\t{svd_text}\t{channel_status}\t{radius_text}\t{note}"
        )
        if self.verbose:
            print(line)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
