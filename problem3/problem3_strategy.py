"""问题 3 的唯一策略文件（供 ``problem3/problem3_main.py`` 运行，也可直接本地随机测试）。

结构：

1. **题面常量与动作日志** —— 题面常量与离线仿真器统一由根目录 ``simulator.py``
   提供（问题 3 / 问题 4 共用），本模块只保留 ``ActionLogger``；
2. **模型层**（``load_implementation()`` 内嵌）—— 本问题**唯一**的策略实现（原
   ``q3_practice_v2.py`` 单文件打包）：内嵌 client / problem3_geometry /
   ``_q3_model_base`` / coverage_strategy / lookahead_strategy。内嵌代码一律以私有
   模块名注册，绝不覆盖本模块 ``problem3_strategy``，以免影响 problem4 与
   ``simulator``；
3. **策略层** —— ``build_empty_channel_strategy`` / ``build_hex_cover_strategy`` /
   ``build_strategy``；``build_strategy()`` 不带开关时即最终演练采用的组合；
4. **本地随机测试** —— ``python problem3/problem3_strategy.py --cases 20``（离线，不联网）。

联网运行：``python problem3/problem3_main.py``；演练或正式由模拟器界面当前选择决定。
"""


from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

# 让 `python problem3/problem3_strategy.py` 也能找到根目录的 simulator.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from problem3_geometry import nearest_neighbor_route, two_opt_open  # noqa: E402

# 题面常量与离线仿真器见根目录 simulator.py（问题 3 / 问题 4 共用）
from simulator import (  # noqa: E402
    BEARING_ERROR_DEG,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_FAIL_TIME_S,
    CLEAR_RADIUS,
    CLEAR_SUCCESS_TIME_S,
    LocalSimulator,
    MAX_RECEIVE_RADIUS,
    MEASURE_TIME_S,
    MIN_RECEIVE_RADIUS,
    MOVE_SPEED_MPS,
    NEAR_RADIUS,
    SOURCE_COUNT_MAX,
    SOURCE_COUNT_MIN,
    SWITCH_TIME_S,
    TARGET_RADIUS,
)

STATUS_UNKNOWN = "UNKNOWN"
STATUS_DETECTED = "DETECTED"
STATUS_CLEARED = "CLEARED"
STATUS_EMPTY = "EMPTY_CERTIFIED"


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
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


import argparse
import json
import sys
from pathlib import Path

from types import ModuleType

def _embedded_module(name, source):
    module = ModuleType(name)
    module.__file__ = __file__
    sys.modules[name] = module
    exec(compile(source, str(__file__) + "::" + name, "exec"), module.__dict__)

def load_implementation():
    # 原始模块：client.py
    _embedded_module('client', r'''"""机器狗与无线电干扰源环境模拟器的通信客户端。

把附件 2《模拟器通信接口说明及编程指南》里的 HTTP+JSON 协议封装成 4 个语义方法：
enter() / measure() / clear() / exit()，并在本地维护机器狗自身的状态：

    - 当前位置 current_position（初始 (0, 0)）
    - 测向机当前频道 current_channel（初始 1，只被 /measure 更新）
    - 虚拟时刻 virtual_time_s（只被 /measure、/clear 推进）
    - 现实剩余时间 remaining_real_s（由 /enter 返回的 remaining_real_duration_s 推算）

协议要点（容易踩坑，这里已全部处理）：
    - 每个新动作必须用新的 request_id，本客户端用自增序号保证唯一，避免 HTTP 409。
    - 只有网络中断/连接失败重试时，才复用原请求内容和原 request_id；4xx/5xx 按协议处理。
    - /clear 的 channel 只是"要清除的目标频道"，不会切换测向机频道、不产生 1 秒耗时。
    - accepted=false 时动作不生效、虚拟时钟不推进，此时响应里的 virtual_time_s 是 0，
      不能当作当前虚拟时刻使用。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

# --------------------------------------------------------------------------- #
# 常量：模拟器默认参数（与附件 2 一致，可在模拟器中修改端口）
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_ARENA_ID = "default"

CHANNEL_MIN = 1
CHANNEL_MAX = 20
COORD_ABS_LIMIT = 2_000_000

MOVE_SPEED_MPS = 5.0        # 移动速度 5 m/s
SWITCH_CHANNEL_S = 1.0      # 任意两频道之间切换 1 秒
MEASURE_ACTION_S = 5.0      # 检测动作 5 秒
CLEAR_NOT_FOUND_S = 3.0     # 光学精确定位未发现目标 3 秒
CLEAR_SUCCESS_S = 5.0       # 定位 + 激光清除 5 秒

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_INTERVAL_S = 1.0

# 日志默认写到项目根目录下的 logs/，避免受当前工作目录影响
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class RobotClientError(Exception):
    """所有客户端异常的基类。"""


class RequestRejected(RobotClientError):
    """HTTP 200 但 accepted=false：动作未生效，虚拟时钟不推进。

    常见原因：arena_id/robot_id 不匹配、存在未声明字段、尚未 /enter、
    重复 /enter、测试已结束。这类请求不占用 request_id，修正后可复用。
    """

    def __init__(self, path: str, response: Dict):
        self.path = path
        self.response = response
        super().__init__(f"{path} 被拒绝：{json.dumps(response, ensure_ascii=False)}")


class HttpError(RobotClientError):
    """HTTP 非 200：400/404/405/409/413/415/429/500。响应体仍是 JSON。"""

    def __init__(self, path: str, status_code: int, response: Optional[Dict]):
        self.path = path
        self.status_code = status_code
        self.response = response or {}
        super().__init__(
            f"{path} HTTP {status_code}：{json.dumps(self.response, ensure_ascii=False)}"
        )


class TransportError(RobotClientError):
    """重试耗尽仍拿不到合法 JSON 响应。

    通常是倒计时未结束、接口尚未开放、测试已结束或网络故障——此时连接可能被
    直接关闭，没有响应体，属于正常现象。
    """


class NotInSessionError(RobotClientError):
    """尚未成功调用 enter() 就试图发送动作。"""


# --------------------------------------------------------------------------- #
# 返回结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeasureResult:
    """一次 /measure 的结果。"""

    result: str                     # "direction" | "near" | "no_signal"
    svd_deg: Optional[float]        # 只有 result == "direction" 时非 None
    position: Tuple[float, float]
    channel: int
    virtual_time_s: float           # 本次检测完成后的虚拟时刻
    raw: Dict = field(default_factory=dict, repr=False)

    @property
    def has_signal(self) -> bool:
        return self.result == "direction"

    @property
    def is_near(self) -> bool:
        """距离干扰源不超过 5 米，信号过强拿不到示向度，可直接去清。"""
        return self.result == "near"


@dataclass(frozen=True)
class ClearResult:
    """一次 /clear 的结果。"""

    cleared: bool                   # True = success，False = no_target_in_range
    position: Tuple[float, float]
    channel: int
    virtual_time_s: float
    raw: Dict = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class RobotClient:
    """机器狗接口客户端。典型用法：

        with RobotClient(ROBOT_ID) as robot:
            enter = robot.enter()
            res = robot.measure(300, 400, 1)
            if res.has_signal:
                print(res.svd_deg)
            robot.exit()
    """

    def __init__(
        self,
        robot_id: str,
        base_url: str = DEFAULT_BASE_URL,
        arena_id: str = DEFAULT_ARENA_ID,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_interval: float = DEFAULT_RETRY_INTERVAL_S,
        log_dir: str = LOG_DIR,
        verbose: bool = False,
    ):
        if not robot_id:
            raise ValueError("robot_id 不能为空，必须是当前登录的参赛队号")
        self.robot_id = str(robot_id)
        self.base_url = base_url.rstrip("/")
        self.arena_id = arena_id
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_interval = retry_interval
        self.verbose = verbose

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # ---- 机器狗状态 ----
        self._seq = 0
        self._in_session = False
        self._current_position: Tuple[float, float] = (0.0, 0.0)
        self._current_channel: Optional[int] = None
        self._virtual_time_s = 0.0
        self._max_virtual_duration_s: Optional[float] = None
        self._max_real_duration_s: Optional[float] = None
        self._real_deadline: Optional[float] = None

        # ---- 日志 ----
        self.log_dir = log_dir
        self.log_path: Optional[str] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(log_dir, f"robot_{stamp}.txt")

    # ---------------------------------------------------------------- 上下文
    def __enter__(self) -> "RobotClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------ 状态
    @property
    def in_session(self) -> bool:
        return self._in_session

    @property
    def current_position(self) -> Tuple[float, float]:
        return self._current_position

    @property
    def current_channel(self) -> Optional[int]:
        """测向机当前频道。只有成功的 /measure 会更新它，/clear 不会。"""
        return self._current_channel

    @property
    def virtual_time_s(self) -> float:
        return self._virtual_time_s

    @property
    def max_virtual_duration_s(self) -> Optional[float]:
        return self._max_virtual_duration_s

    @property
    def max_real_duration_s(self) -> Optional[float]:
        return self._max_real_duration_s

    @property
    def remaining_real_s(self) -> float:
        """本局还能用的现实时间（秒）。未 enter 时返回 0。"""
        if self._real_deadline is None:
            return 0.0
        return max(0.0, self._real_deadline - time.monotonic())

    def is_time_up(self, margin_s: float = 0.0) -> bool:
        """现实时间或虚拟时间是否即将/已经耗尽。

        margin_s 是安全余量：留出收尾（比如最后几次 /clear 和 /exit）所需的时间。
        """
        if self._real_deadline is not None and self.remaining_real_s <= margin_s:
            return True
        if (
            self._max_virtual_duration_s is not None
            and self._virtual_time_s + margin_s >= self._max_virtual_duration_s
        ):
            return True
        return False

    # ------------------------------------------------------------ 耗时预估
    def estimate_move_time(self, x: float, y: float) -> float:
        """从当前位置直线移动到 (x, y) 的耗时（秒）。"""
        cx, cy = self._current_position
        return math.hypot(x - cx, y - cy) / MOVE_SPEED_MPS

    def estimate_measure_cost(self, x: float, y: float, channel: int) -> float:
        """一次 /measure 的虚拟耗时：移动 + 切频道(0/1) + 5。"""
        switch = 0.0
        if self._current_channel is not None and channel != self._current_channel:
            switch = SWITCH_CHANNEL_S
        return self.estimate_move_time(x, y) + switch + MEASURE_ACTION_S

    def estimate_clear_cost(self, x: float, y: float, found: bool = True) -> float:
        """一次 /clear 的虚拟耗时：移动 + 3(未发现) 或 5(成功)。不含切频道。"""
        action = CLEAR_SUCCESS_S if found else CLEAR_NOT_FOUND_S
        return self.estimate_move_time(x, y) + action

    # ---------------------------------------------------------------- 日志
    def _log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        if self.verbose:
            print(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # ------------------------------------------------------------ 参数校验
    @staticmethod
    def _check_position(x: float, y: float) -> Tuple[float, float]:
        for name, value in (("x", x), ("y", y)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"position.{name} 必须是有限数值，收到 {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"position.{name} 不能是 NaN 或无穷大")
            if abs(value) > COORD_ABS_LIMIT:
                raise ValueError(f"position.{name} 绝对值不得超过 {COORD_ABS_LIMIT}")
        return (float(x), float(y))

    @staticmethod
    def _check_channel(channel: int) -> int:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ValueError(f"channel 必须是 {CHANNEL_MIN}..{CHANNEL_MAX} 的整数，收到 {channel!r}")
        if isinstance(channel, float) and not channel.is_integer():
            raise ValueError(f"channel 必须是整数，收到 {channel!r}")
        channel = int(channel)
        if not CHANNEL_MIN <= channel <= CHANNEL_MAX:
            raise ValueError(f"channel 必须在 {CHANNEL_MIN}..{CHANNEL_MAX} 之间，收到 {channel}")
        return channel

    def _next_request_id(self, prefix: str) -> str:
        """每个新动作一个全新 request_id，避免同一 ID 对应不同动作导致 409。"""
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def _base_payload(self, prefix: str) -> Dict:
        return {
            "arena_id": self.arena_id,
            "robot_id": self.robot_id,
            "request_id": self._next_request_id(prefix),
        }

    # -------------------------------------------------------------- 通信核心
    def _post(self, path: str, payload: Dict) -> Dict:
        """发送一次请求。

        - 网络异常 / 无 JSON 体 / 429 / 500：复用完全相同的 payload 与 request_id
          重试（协议规定重试必须复用原 ID，模拟器保证幂等）。
        - 其余 4xx：不重试，直接抛 HttpError。
        - HTTP 200 但 accepted=false：抛 RequestRejected。
        """
        url = self.base_url + path

        for attempt in range(1, self.max_retries + 1):
            self._log(f"REQUEST attempt={attempt} path={path} payload={json.dumps(payload, ensure_ascii=False)}")

            try:
                response = self._session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as error:
                self._log(f"ERROR attempt={attempt} path={path} error={repr(error)}")
                if attempt == self.max_retries:
                    raise TransportError(
                        f"{path} 连续 {self.max_retries} 次请求失败（接口未开放/测试已结束/网络故障）：{error!r}"
                    ) from error
                time.sleep(self.retry_interval)
                continue

            try:
                data = response.json()
            except ValueError as error:
                self._log(
                    f"ERROR attempt={attempt} path={path} invalid_json={repr(error)} text={response.text}"
                )
                if attempt == self.max_retries:
                    raise TransportError(
                        f"{path} 连续 {self.max_retries} 次未取得合法 JSON 响应"
                    ) from error
                time.sleep(self.retry_interval)
                continue

            self._log(f"RESPONSE attempt={attempt} status={response.status_code} json={json.dumps(data, ensure_ascii=False)}")

            # 429（限流保护）/ 500（模拟器内部错误）：同 ID 重试是安全的
            if response.status_code in (429, 500) and attempt < self.max_retries:
                time.sleep(self.retry_interval * attempt)
                continue

            if response.status_code != 200:
                raise HttpError(path, response.status_code, data)

            if data.get("accepted") is not True:
                raise RequestRejected(path, data)

            return data

        raise TransportError(f"{path} 请求失败且已重试 {self.max_retries} 次")

    # -------------------------------------------------------------- 四条指令
    def enter(self) -> Dict:
        """POST /enter：进入目标区域，开始计时。指令本身不推进虚拟时钟。

        返回完整响应，其中 remaining_real_duration_s 是本局实际可用现实秒数，
        必须用这个值做倒计时，不能假定固定 1200 秒。
        """
        data = self._post("/enter", self._base_payload("enter"))

        self._in_session = True
        self._current_position = (0.0, 0.0)
        self._current_channel = 1
        self._virtual_time_s = float(data.get("virtual_time_s", 0.0))
        self._max_virtual_duration_s = data.get("max_virtual_duration_s")
        self._max_real_duration_s = data.get("max_real_duration_s")
        remaining = float(data.get("remaining_real_duration_s", 0.0))
        self._real_deadline = time.monotonic() + remaining
        return data

    def measure(self, x: float, y: float, channel: int) -> MeasureResult:
        """POST /measure：移动到 (x, y) 并对指定频道检测。

        虚拟耗时 = 移动(d/5) + 切频道(0 或 1) + 5。
        """
        if not self._in_session:
            raise NotInSessionError("必须先成功调用 enter() 才能检测")
        position = self._check_position(x, y)
        channel = self._check_channel(channel)

        payload = self._base_payload("measure")
        payload["position"] = {"x": position[0], "y": position[1]}
        payload["channel"] = channel

        data = self._post("/measure", payload)

        result = data.get("measure_result")
        svd = data.get("svd_deg")
        if result == "direction" and svd is None:
            # 协议保证 direction 必带 svd_deg，这里只做防御
            raise RobotClientError(f"/measure 返回 direction 但缺少 svd_deg：{data}")

        # 只有被接受的动作才更新状态
        self._virtual_time_s = float(data.get("virtual_time_s", self._virtual_time_s))
        self._current_position = position
        self._current_channel = channel

        return MeasureResult(
            result=result,
            svd_deg=svd,
            position=position,
            channel=channel,
            virtual_time_s=self._virtual_time_s,
            raw=data,
        )

    def clear(self, x: float, y: float, channel: int) -> ClearResult:
        """POST /clear：移动到 (x, y)，清除 20 米内指定频道的干扰源。

        注意：channel 只用于指定要清除的目标，**不会切换测向机频道**，也不产生
        1 秒切换耗时。虚拟耗时 = 移动(d/5) + 3(未发现) 或 5(成功清除)。
        """
        if not self._in_session:
            raise NotInSessionError("必须先成功调用 enter() 才能清除")
        position = self._check_position(x, y)
        channel = self._check_channel(channel)

        payload = self._base_payload("clear")
        payload["position"] = {"x": position[0], "y": position[1]}
        payload["channel"] = channel

        data = self._post("/clear", payload)

        cleared = data.get("clear_result") == "success"

        self._virtual_time_s = float(data.get("virtual_time_s", self._virtual_time_s))
        self._current_position = position
        # 关键：/clear 不改变 self._current_channel

        return ClearResult(
            cleared=cleared,
            position=position,
            channel=channel,
            virtual_time_s=self._virtual_time_s,
            raw=data,
        )

    def exit(self) -> str:
        """POST /exit：主动结束测试。返回 exit_reason（正常为 "user_exit"）。"""
        if not self._in_session:
            raise NotInSessionError("尚未 enter()，无需 exit()")
        data = self._post("/exit", self._base_payload("exit"))
        self._in_session = False
        return data.get("exit_reason")
''')
    # 原始模块：problem3_geometry.py
    _embedded_module('problem3_geometry', r'''"""问题 3 几何工具库。

本模块只做“保守”的几何近似，核心不变量是：

    **程序保存的可行域永远包含干扰源的真实位置。**

因此所有会引入误差的操作都朝“把区域放大”的方向取近似，具体约定：

* 圆用**外切正多边形**表示（多边形 ⊇ 圆），求交后区域只会比真实可行域大；
* 方向扇形写成两个半平面，用 Sutherland-Hodgman 对凸多边形裁剪；
* 排除圆（`no_signal` 排除的 1000 m、清除失败排除的 20 m）只记录，
  不参与凸多边形运算——排除只会“缩小”可行域，忽略它保证包含性，最多多算几步；
* 最小包围圆对凸多边形用确定性 Welzl 增量算法，结果一定覆盖多边形。

坐标一律为 numpy `(2,)` 或 `(N, 2)` 数组，单位米；角度参数用弧度还是度在函数名/文档里写明。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

TAU = 2.0 * math.pi

Point = np.ndarray
Polygon = np.ndarray

# 裁剪容差：坐标量级 ~1e3，1e-9 的绝对容差足够判断“点在边界上”
_CLIP_EPS = 1e-9
# 多边形去重容差
_DEDUP_EPS = 1e-7


# --------------------------------------------------------------------------- #
# 基础向量
# --------------------------------------------------------------------------- #
def unit_from_deg(angle_deg: float) -> Point:
    """由角度（度）生成单位向量。"""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def unit_from_rad(angle_rad: float) -> Point:
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def unit(v: Sequence[float]) -> Point:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def perpendicular_left(v: Sequence[float]) -> Point:
    """逆时针旋转 90°。"""
    v = np.asarray(v, dtype=float)
    return np.array([-v[1], v[0]], dtype=float)


def wrap_angle_deg(angle_deg: float) -> float:
    """把角度规整到 [0, 360)。"""
    return angle_deg % 360.0


def signed_angle_diff_deg(a: float, b: float) -> float:
    """a - b 规整到 (-180, 180]，用于比较示向度。"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


# --------------------------------------------------------------------------- #
# 圆与多边形
# --------------------------------------------------------------------------- #
def circumscribed_polygon(center: Sequence[float], radius: float, n: int = 128) -> Polygon:
    """半径 radius 的圆的外切正 n 边形（多边形包含整个圆）。

    顶点到圆心距离为 radius / cos(pi/n)，n 越大越贴合圆。
    """
    center = np.asarray(center, dtype=float)
    factor = 1.0 / math.cos(math.pi / n)
    ang = TAU * np.arange(n) / n
    ring = factor * radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return center + ring


def clip_halfplane(poly: Polygon, a: float, b: float, c: float) -> Polygon:
    """用半平面 ``a*x + b*y + c >= 0`` 裁剪凸多边形（Sutherland-Hodgman）。"""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2))

    out: List[Point] = []
    m = len(poly)
    for i in range(m):
        p = poly[i]
        q = poly[(i + 1) % m]
        fp = a * p[0] + b * p[1] + c
        fq = a * q[0] + b * q[1] + c
        p_in = fp >= -_CLIP_EPS
        q_in = fq >= -_CLIP_EPS
        if p_in:
            out.append(p)
        if p_in != q_in:
            denom = fp - fq
            if abs(denom) > 1e-15:
                t = fp / denom
                out.append(p + t * (q - p))
    return _clean_polygon(out)


def clip_wedge(poly: Polygon, apex: Sequence[float], theta_rad: float,
               half_rad: float) -> Polygon:
    """保留以 apex 为顶点、沿 theta_rad、张角 ±half_rad 的**前向射线扇形**。

    扇形 = 两个半平面的交，天然是“从测点向前延伸”的射线扇形，
    而不是穿过测点的无限直线带。
    """
    apex = np.asarray(apex, dtype=float)
    u_minus = unit_from_rad(theta_rad - half_rad)
    u_plus = unit_from_rad(theta_rad + half_rad)

    # cross(u_minus, P-apex) >= 0
    poly = clip_halfplane(
        poly,
        -u_minus[1], u_minus[0],
        u_minus[1] * apex[0] - u_minus[0] * apex[1],
    )
    # cross(u_plus, P-apex) <= 0  ⇔  -cross(u_plus, P-apex) >= 0
    poly = clip_halfplane(
        poly,
        u_plus[1], -u_plus[0],
        -(u_plus[1] * apex[0] - u_plus[0] * apex[1]),
    )
    return poly


def intersect_convex(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """两个逆时针凸多边形的交（用 poly_b 的每条边裁剪 poly_a）。"""
    if poly_a is None or poly_b is None or len(poly_a) == 0 or len(poly_b) == 0:
        return np.empty((0, 2))

    out = poly_a
    m = len(poly_b)
    for i in range(m):
        p = poly_b[i]
        q = poly_b[(i + 1) % m]
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        # 逆时针多边形内部在边左侧：cross((dx,dy), X-p) >= 0
        out = clip_halfplane(out, -dy, dx, dy * p[0] - dx * p[1])
        if len(out) == 0:
            return out
    return out


def intersect_disk(poly: Polygon, center: Sequence[float], radius: float,
                   n: int = 128) -> Polygon:
    """poly 与盘(center, radius)相交，盘用外切正多边形保守表示。"""
    return intersect_convex(poly, circumscribed_polygon(center, radius, n))


def _clean_polygon(points: Sequence[Point]) -> Polygon:
    """去掉相邻重复点，并把点列表转成 (N,2) 数组。"""
    if not points:
        return np.empty((0, 2))
    kept: List[Point] = []
    for p in points:
        if not kept or math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) > _DEDUP_EPS:
            kept.append(p)
    if len(kept) >= 2:
        first, last = kept[0], kept[-1]
        if math.hypot(first[0] - last[0], first[1] - last[1]) <= _DEDUP_EPS:
            kept.pop()
    if not kept:
        return np.empty((0, 2))
    return np.asarray(kept, dtype=float)


def polygon_area(poly: Polygon) -> float:
    """多边形面积（逆时针为正）。"""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: Polygon) -> Point:
    if poly is None or len(poly) == 0:
        raise ValueError("空多边形没有质心")
    if len(poly) < 3:
        return poly.mean(axis=0)
    x = poly[:, 0]
    y = poly[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = float(cross.sum())
    if abs(area2) < 1e-12:
        return poly.mean(axis=0)
    cx = float(((x + x1) * cross).sum()) / (3.0 * area2)
    cy = float(((y + y1) * cross).sum()) / (3.0 * area2)
    return np.array([cx, cy], dtype=float)


# --------------------------------------------------------------------------- #
# 最小包围圆（确定性 Welzl 增量算法）
# --------------------------------------------------------------------------- #
def _dist_to(pt: Point, cx: float, cy: float) -> float:
    return math.hypot(pt[0] - cx, pt[1] - cy)


def _circle_from_two(p: Point, q: Point) -> Tuple[float, float, float]:
    cx = 0.5 * (p[0] + q[0])
    cy = 0.5 * (p[1] + q[1])
    return cx, cy, math.hypot(p[0] - q[0], p[1] - q[1]) * 0.5


def _circle_from_three(p: Point, q: Point, r: Point) -> Tuple[float, float, float]:
    ax, ay = float(p[0]), float(p[1])
    bx, by = float(q[0]), float(q[1])
    cx, cy = float(r[0]), float(r[1])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        # 三点共线：退化为覆盖这三点所需的最小圆（由最远两点决定）
        best = _circle_from_two(p, q)
        best = min(
            [best, _circle_from_two(p, r), _circle_from_two(q, r)],
            key=lambda c: c[2],
        )
        cands = [best, _circle_from_two(p, q), _circle_from_two(p, r), _circle_from_two(q, r)]
        valid = [c for c in cands if _dist_to(p, c[0], c[1]) <= c[2] + 1e-7
                 and _dist_to(q, c[0], c[1]) <= c[2] + 1e-7
                 and _dist_to(r, c[0], c[1]) <= c[2] + 1e-7]
        return min(valid, key=lambda c: c[2])

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def min_enclosing_circle(points: Sequence[Point]) -> Optional[Tuple[Point, float]]:
    """返回覆盖全部点的最小圆 (center, radius)；无点返回 None。"""
    if points is None or len(points) == 0:
        return None
    pts = [(float(p[0]), float(p[1])) for p in points]

    cx = cy = 0.0
    radius = -1.0
    for i, p in enumerate(pts):
        if radius < 0 or _dist_to(p, cx, cy) > radius + 1e-7:
            cx, cy, radius = p[0], p[1], 0.0
            for j in range(i):
                q = pts[j]
                if _dist_to(q, cx, cy) > radius + 1e-7:
                    cx, cy, radius = _circle_from_two(p, q)
                    for k in range(j):
                        r = pts[k]
                        if _dist_to(r, cx, cy) > radius + 1e-7:
                            cx, cy, radius = _circle_from_three(p, q, r)
    return np.array([cx, cy], dtype=float), float(radius)


def farthest_pair(poly: Polygon) -> Tuple[Point, float]:
    """凸多边形上距离最远的一对顶点：返回 (单位方向, 距离)。"""
    if poly is None or len(poly) < 2:
        return np.array([1.0, 0.0]), 0.0
    best_d = -1.0
    best = (poly[0], poly[0])
    pts = poly
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > best_d:
                best_d = d
                best = (pts[i], pts[j])
    direction = best[1] - best[0]
    if math.hypot(direction[0], direction[1]) < 1e-12:
        return np.array([1.0, 0.0]), 0.0
    return unit(direction), float(best_d)


# --------------------------------------------------------------------------- #
# 网格覆盖与路径
# --------------------------------------------------------------------------- #
def grid_cover_points(poly: Polygon, spacing: float,
                      inside_only: bool = False) -> List[Tuple[float, float]]:
    """生成覆盖 poly 包围盒的方格点（逐行蛇形顺序），保证覆盖整个包围盒。

    spacing 取 ``20*sqrt(2)*(1-eps)`` 时，包围盒内任意点到最近方格点距离严格小于 20 m。
    ``inside_only=False`` 时返回包围盒内全部方格点——这样无论可行域形状如何，
    覆盖性都由“包围盒被完整覆盖”保证。
    """
    if poly is None or len(poly) == 0:
        return []
    xs_min, ys_min = poly.min(axis=0)
    xs_max, ys_max = poly.max(axis=0)

    nx = int(math.floor((xs_max - xs_min) / spacing)) + 1
    ny = int(math.floor((ys_max - ys_min) / spacing)) + 1
    xs = xs_min + spacing * np.arange(nx + 1)
    ys = ys_min + spacing * np.arange(ny + 1)
    xs[-1] = max(xs[-1], xs_max)
    ys[-1] = max(ys[-1], ys_max)

    pts: List[Tuple[float, float]] = []
    for iy, y in enumerate(ys):
        row = xs if iy % 2 == 0 else xs[::-1]
        for x in row:
            if inside_only and not point_in_polygon((x, y), poly):
                continue
            pts.append((float(x), float(y)))
    return pts


def point_in_polygon(pt: Sequence[float], poly: Polygon) -> bool:
    """射线法判断点是否在（凸）多边形内或边界上。"""
    if poly is None or len(poly) < 3:
        return False
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def distance_point_to_polygon(pt: Sequence[float], poly: Polygon) -> float:
    """点到多边形（边界与内部）的距离，点在内部返回 0。"""
    if poly is None or len(poly) == 0:
        return float("inf")
    if point_in_polygon(pt, poly):
        return 0.0
    px, py = float(pt[0]), float(pt[1])
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            d = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        best = min(best, d)
    return best


def nearest_neighbor_route(start: Sequence[float],
                           points: Sequence[Sequence[float]]) -> List[int]:
    """最近邻顺序，返回 points 的下标排列。"""
    n = len(points)
    if n == 0:
        return []
    remaining = list(range(n))
    route: List[int] = []
    cur = np.asarray(start, dtype=float)
    while remaining:
        idx = min(
            remaining,
            key=lambda i: math.hypot(points[i][0] - cur[0], points[i][1] - cur[1]),
        )
        route.append(idx)
        cur = np.asarray(points[idx], dtype=float)
        remaining.remove(idx)
    return route


def route_length(route: Sequence[int], start: Sequence[float],
                 points: Sequence[Sequence[float]]) -> float:
    if not route:
        return 0.0
    total = 0.0
    cur = np.asarray(start, dtype=float)
    for i in route:
        nxt = np.asarray(points[i], dtype=float)
        total += math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        cur = nxt
    return total


def two_opt_open(route: List[int], start: Sequence[float],
                 points: Sequence[Sequence[float]]) -> List[int]:
    """开放路径 2-opt：不要求回到起点。"""
    if len(route) < 3:
        return route
    best = route[:]
    best_len = route_length(best, start, points)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                candidate = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                cand_len = route_length(candidate, start, points)
                if cand_len + 1e-9 < best_len:
                    best, best_len = candidate, cand_len
                    improved = True
    return best
''')
    # 原始模块：problem3_strategy.py
    _embedded_module('_q3_model_base', r'''"""问题 3：机器狗自动搜索、定位与清除策略。

字典序目标：

1. 保证所有合法情况下都不漏检，并最终清除全部干扰源；
2. 在满足第 1 项的策略中，尽量减小“均匀面积分布”下的平均虚拟时间。

策略层只依赖一个“类 RobotClient 接口”的对象（鸭子类型），因此既能接真实
``client.RobotClient``，也能接离线 ``simulator.LocalSimulator``。用到的成员：

    enter() / measure(x, y, channel) / clear(x, y, channel) / exit()
    current_position / current_channel / virtual_time_s / remaining_real_s
    is_time_up(margin_s) / estimate_move_time(x, y) / in_session

本模块不做任何 HTTP，可在没有模拟器时用本地仿真完整跑通。
"""

from __future__ import annotations

import math
import os
from itertools import combinations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from problem3_geometry import (
    clip_halfplane,
    clip_wedge,
    circumscribed_polygon,
    farthest_pair,
    grid_cover_points,
    intersect_convex,
    intersect_disk,
    min_enclosing_circle,
    nearest_neighbor_route,
    perpendicular_left,
    point_in_polygon,
    polygon_area,
    two_opt_open,
    unit_from_deg,
)

# --------------------------------------------------------------------------- #
# 题面常量
# --------------------------------------------------------------------------- #
TARGET_RADIUS = 1800.0          # 目标圆域半径 m
MIN_RECEIVE_RADIUS = 1000.0     # 干扰源最小有效接收半径 m
MAX_RECEIVE_RADIUS = 1500.0     # 干扰源最大有效接收半径 m
NEAR_RADIUS = 5.0               # 返回 near 的距离阈值 m
CLEAR_RADIUS = 20.0             # 清除半径 m
BEARING_ERROR_DEG = 1.0         # 示向度最大绝对误差 度
MOVE_SPEED_MPS = 5.0            # 移动速度 m/s
MEASURE_TIME_S = 5.0            # 单次检测时间 s
SWITCH_TIME_S = 1.0             # 切换检测频道时间 s
CLEAR_FAIL_TIME_S = 3.0         # 清除失败操作时间 s
CLEAR_SUCCESS_TIME_S = 5.0      # 清除成功操作时间 s
CHANNEL_MIN = 1
CHANNEL_MAX = 20
SOURCE_COUNT_MIN = 10
SOURCE_COUNT_MAX = 16
SURVEY_RING_RADIUS = 1150.0     # 六边形巡检点半径 m
SAFE_REGION_RADIUS = 58.0       # 两次清除法可行域半径上界 m

# 策略内部可调参数
GRID_SPACING = CLEAR_RADIUS * math.sqrt(2.0) * (1.0 - 1e-3)  # 保证网格覆盖 20 m 圆
PROBE_R_EST_MIN = 100.0         # 动态拉偏：目标距离估计下限
PROBE_R_EST_MAX = 800.0         # 动态拉偏：目标距离估计上限
PROBE_SIDE_RATIO = 0.6          # 动态拉偏：侧向偏移比例，兼顾交会角与移动距离
PROBE_SIDE_MAX = 350.0          # 动态拉偏：侧向偏移上限
PROBE_FORWARD_RATIO = 0.5       # 动态拉偏：前向跟进比例
PROBE_BACKUP_FORWARD = 350.0    # 短距离备用补测点，避免动态拉偏失败后过早大步长外跑
PROBE_BACKUP_SIDE = 200.0
PROBE_FAR_BACKUP_FORWARD = 600.0  # 全距离接收保证备用补测点
PROBE_FAR_BACKUP_SIDE = 300.0
PROBE_STAGNATION_LIMIT = 3      # 连续多少次补测未缩小可行域就转网格兜底
PROBE_MAX_PER_CHANNEL = 12      # 单频道补测次数上限
REGION_CLUSTER_DISTANCE = 150.0  # 可行域中心小于该距离则归为同一空间批次
OPPORTUNISTIC_MAX_DETOUR = 120.0  # 顺路清除允许增加的最大路程 m
TSP_EXACT_LIMIT = 9          # Held-Karp 精确开放路径的节点上限

STATUS_UNKNOWN = "UNKNOWN"
STATUS_DETECTED = "DETECTED"
STATUS_CLEARED = "CLEARED"
STATUS_EMPTY = "EMPTY_CERTIFIED"


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
@dataclass
class ChannelState:
    channel: int
    status: str = STATUS_UNKNOWN
    # 可行域：凸多边形（保守包含真实位置）
    feasible: Optional[np.ndarray] = None
    # 排除圆列表 (cx, cy, r)，只做记录与诊断
    exclusions: List[Tuple[float, float, float]] = field(default_factory=list)
    # 最小包围圆 (center(np.array), radius)
    enclosing: Optional[Tuple[np.ndarray, float]] = None
    # 第一次测向（位置、单位方向），用于第一组补测点
    first_direction: Optional[Tuple[np.ndarray, np.ndarray]] = None
    # 补测计划（第一组两个点，先近后远）
    probe_plan: List[np.ndarray] = field(default_factory=list)
    probe_count: int = 0
    stagnation: int = 0
    grid_mode: bool = False
    last_probe: Optional[np.ndarray] = None
    completed_points: Set[int] = field(default_factory=set)

    @property
    def radius(self) -> Optional[float]:
        return None if self.enclosing is None else float(self.enclosing[1])

    @property
    def center(self) -> Optional[np.ndarray]:
        return None if self.enclosing is None else self.enclosing[0]


@dataclass
class StrategyConfig:
    target_radius: float = TARGET_RADIUS
    survey_radius: float = SURVEY_RING_RADIUS
    bearing_error_deg: float = BEARING_ERROR_DEG
    safe_region_radius: float = SAFE_REGION_RADIUS
    time_margin_s: float = 30.0          # 现实时间安全余量
    verbose: bool = False
    log_path: Optional[str] = None
    # 达到 16 个干扰源后可直接结束
    stop_at_source_count: int = SOURCE_COUNT_MAX
    # Region Clustering：按可行域中心聚类，批次内连续局部化/清除，减少跨区域往返
    region_cluster_distance: float = REGION_CLUSTER_DISTANCE
    # Opportunistic Strike：移动途中若小半径目标几乎顺路，则立即清除
    opportunistic_max_detour: float = OPPORTUNISTIC_MAX_DETOUR
    opportunistic_enabled: bool = True


@dataclass
class RouteTask:
    kind: str
    point: np.ndarray
    channel_state: ChannelState
    note: str = ""


# --------------------------------------------------------------------------- #
# 七点巡检点
# --------------------------------------------------------------------------- #
def build_survey_points(radius: float = SURVEY_RING_RADIUS) -> List[np.ndarray]:
    """P0 = 原点，P1..P6 = 半径 radius 的正六边形顶点（角度 0,60,...,300°）。"""
    pts = [np.array([0.0, 0.0])]
    for k in range(6):
        a = math.radians(60.0 * k)
        pts.append(np.array([radius * math.cos(a), radius * math.sin(a)]))
    return pts


# --------------------------------------------------------------------------- #
# 策略主体
# --------------------------------------------------------------------------- #
class Problem3Strategy:
    def __init__(self, robot, config: Optional[StrategyConfig] = None):
        self.robot = robot
        self.cfg = config or StrategyConfig()
        self.logger = ActionLogger(self.cfg.log_path, self.cfg.verbose)

        self.channels: Dict[int, ChannelState] = {
            c: ChannelState(channel=c) for c in range(CHANNEL_MIN, CHANNEL_MAX + 1)
        }
        self.seq = 0
        self.cleared_count = 0
        self._session_owned = False

        # 统计
        self.stats: Dict[str, float] = {
            "measures": 0, "clears": 0, "clear_success": 0,
            "move_distance": 0.0, "switch_count": 0,
            "probes": 0, "grid_clears": 0,
            "two_clear_used": 0, "grid_channels": 0,
            "clusters": 0, "opportunistic_strikes": 0,
            "shared_measures": 0,
            "empty_pruned": 0, "survey_skipped_empty": 0,
            "clipped_to_clear": 0,
        }

    # -------------------------------------------------------------- 运行入口
    def run(self) -> Dict[str, object]:
        self._enter()
        try:
            self._phase_survey()
            self._phase_tsp_routing()
        finally:
            self._exit()
        return self._summary()

    # ------------------------------------------------------------ 底层动作封装
    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _log(self, phase: str, action: str, x: float, y: float, channel: int,
             result: str = "-", svd_deg: Optional[float] = None,
             cs: Optional[ChannelState] = None, note: str = "") -> None:
        self.logger.log(
            seq=self._next_seq(),
            virtual_time=self.robot.virtual_time_s,
            phase=phase, action=action, x=x, y=y, channel=channel,
            result=result, svd_deg=svd_deg,
            channel_status=cs.status if cs else "-",
            feasible_radius=cs.radius if cs else None,
            note=note,
        )

    def _enter(self) -> None:
        if not getattr(self.robot, "in_session", False):
            self.robot.enter()
            self._session_owned = True
        self._log("INIT", "enter", 0.0, 0.0, CHANNEL_MIN,
                  result="entered",
                  note=f"remaining_real_s={self.robot.remaining_real_s:.1f}")

    def _exit(self) -> None:
        try:
            if getattr(self.robot, "in_session", False):
                self.robot.exit()
                self._log("EXIT", "exit", self.robot.current_position[0],
                          self.robot.current_position[1], CHANNEL_MIN,
                          result="exit", note=f"cleared={self.cleared_count}")
        except Exception as error:  # noqa: BLE001 - 收尾失败不应掩盖主流程结果
            self._log("EXIT", "exit_failed", 0.0, 0.0, CHANNEL_MIN,
                      result="error", note=repr(error))

    def _measure(self, x: float, y: float, channel: int, phase: str,
                 cs: Optional[ChannelState] = None, note: str = ""):
        before = self.robot.current_position
        self.stats["move_distance"] += math.hypot(x - before[0], y - before[1])
        cur_ch = self.robot.current_channel
        if cur_ch is not None and cur_ch != channel:
            self.stats["switch_count"] += 1
        res = self.robot.measure(x, y, channel)
        self.stats["measures"] += 1
        self._log(phase, "measure", x, y, channel, result=res.result,
                  svd_deg=res.svd_deg, cs=cs, note=note)
        return res

    def _clear(self, x: float, y: float, channel: int, phase: str,
               cs: Optional[ChannelState] = None, note: str = ""):
        before = self.robot.current_position
        self.stats["move_distance"] += math.hypot(x - before[0], y - before[1])
        res = self.robot.clear(x, y, channel)
        self.stats["clears"] += 1
        if res.cleared:
            self.stats["clear_success"] += 1
        self._log(phase, "clear", x, y, channel,
                  result="success" if res.cleared else "no_target_in_range",
                  cs=cs, note=note)
        return res

    def _time_left(self) -> bool:
        return not self.robot.is_time_up(self.cfg.time_margin_s)

    # ------------------------------------------------------------ 可行域更新
    def _initial_region(self) -> np.ndarray:
        return circumscribed_polygon((0.0, 0.0), self.cfg.target_radius)

    def _update_direction(self, cs: ChannelState, S: np.ndarray,
                          svd_deg: float) -> None:
        """direction：楔形 ∩ 盘(S,1500)，并排除盘(S,5)。"""
        if cs.feasible is None:
            cs.feasible = self._initial_region()
        theta = math.radians(svd_deg)
        half = math.radians(self.cfg.bearing_error_deg + 1e-12)
        cs.feasible = clip_wedge(cs.feasible, S, theta, half)
        cs.feasible = intersect_disk(cs.feasible, S, MAX_RECEIVE_RADIUS)
        cs.exclusions.append((float(S[0]), float(S[1]), NEAR_RADIUS))
        if cs.first_direction is None:
            cs.first_direction = (np.asarray(S, dtype=float),
                                  unit_from_deg(svd_deg))
        self._recompute_enclosing(cs)

    def _update_near(self, cs: ChannelState, S: np.ndarray) -> None:
        """near：盘(S,5)。"""
        if cs.feasible is None:
            cs.feasible = self._initial_region()
        cs.feasible = intersect_disk(cs.feasible, S, NEAR_RADIUS)
        self._recompute_enclosing(cs)

    def _update_no_signal(self, cs: ChannelState, S: np.ndarray) -> None:
        """no_signal：记录排除盘；若此前有信号点，则加入保守线性半平面。"""
        cs.exclusions.append((float(S[0]), float(S[1]), MIN_RECEIVE_RADIUS))
        if cs.feasible is not None and cs.first_direction is not None:
            before_radius = cs.radius
            signal_point, _ = cs.first_direction
            q = np.asarray(S, dtype=float)
            p = np.asarray(signal_point, dtype=float)
            direction = q - p
            # 由 |G-p| <= rc < |G-q| 得 2G·(q-p) <= |q|^2 - |p|^2。
            # clip_halfplane 形式是 a*x+b*y+c >= 0，故整体取负。
            rhs = float(np.dot(q, q) - np.dot(p, p))
            cs.feasible = clip_halfplane(
                cs.feasible,
                -2.0 * float(direction[0]),
                -2.0 * float(direction[1]),
                rhs,
            )
            self._recompute_enclosing(cs)
            if (
                (before_radius is None or before_radius > self.cfg.safe_region_radius)
                and cs.radius is not None
                and cs.radius <= self.cfg.safe_region_radius
            ):
                self.stats["clipped_to_clear"] += 1

    def _update_clear_failure(self, cs: ChannelState, C: np.ndarray) -> None:
        cs.exclusions.append((float(C[0]), float(C[1]), CLEAR_RADIUS))

    def _recompute_enclosing(self, cs: ChannelState) -> None:
        if cs.feasible is None or len(cs.feasible) == 0 or polygon_area(cs.feasible) < 1e-6:
            cs.enclosing = None
            return
        cs.enclosing = min_enclosing_circle(cs.feasible)

    def _region_contains_safe(self, cs: ChannelState) -> bool:
        return cs.feasible is not None and len(cs.feasible) >= 3

    # ---------------------------------------------------------------- 阶段A
    def _channel_order(self, station_id: int) -> List[int]:
        if station_id % 2 == 0:
            return list(range(CHANNEL_MIN, CHANNEL_MAX + 1))
        return list(range(CHANNEL_MAX, CHANNEL_MIN - 1, -1))

    def _known_source_count(self) -> int:
        """已确认存在过干扰源的频道数：DETECTED + CLEARED。"""
        return sum(
            1 for cs in self.channels.values()
            if cs.status in (STATUS_DETECTED, STATUS_CLEARED)
        )

    def _mark_empty(self, cs: ChannelState, note: str) -> None:
        if cs.status == STATUS_UNKNOWN:
            cs.status = STATUS_EMPTY
            self.stats["empty_pruned"] += 1
            self._log("PRUNE", "mark_empty", self.robot.current_position[0],
                      self.robot.current_position[1], cs.channel,
                      result="empty", cs=cs, note=note)

    def _certify_empty_channels(self, stations_count: int) -> None:
        """只执行确定性安全的空频道判定。

        1. 频道完成全部七点检测仍未发现，则由七点覆盖定理判空；
        2. 已确认存在的频道数达到题面上限 16，则剩余 UNKNOWN 频道必为空。
        """
        all_station_ids = set(range(stations_count))
        for cs in self.channels.values():
            if cs.status == STATUS_UNKNOWN and cs.completed_points == all_station_ids:
                self._mark_empty(cs, "all survey stations no_signal")

        if self._known_source_count() >= SOURCE_COUNT_MAX:
            for cs in self.channels.values():
                if cs.status == STATUS_UNKNOWN:
                    self._mark_empty(cs, "source upper bound reached")

    def _should_skip_survey_channel(self, cs: ChannelState,
                                    stations_count: int) -> bool:
        self._certify_empty_channels(stations_count)
        if cs.status in (STATUS_CLEARED, STATUS_EMPTY):
            self.stats["survey_skipped_empty"] += 1
            return True
        if self._known_source_count() >= SOURCE_COUNT_MAX and cs.status == STATUS_UNKNOWN:
            self._mark_empty(cs, "source upper bound reached")
            self.stats["survey_skipped_empty"] += 1
            return True
        return False

    def _phase_survey(self) -> None:
        stations = build_survey_points(self.cfg.survey_radius)
        for station_id, pos in enumerate(stations):
            if not self._time_left():
                break
            if self.cleared_count >= self.cfg.stop_at_source_count:
                break
            for channel in self._channel_order(station_id):
                cs = self.channels[channel]
                if self._should_skip_survey_channel(cs, len(stations)):
                    continue
                if not self._time_left():
                    break
                res = self._measure(float(pos[0]), float(pos[1]), channel,
                                    phase="SURVEY", cs=cs,
                                    note=f"station={station_id}")
                cs.completed_points.add(station_id)

                if res.result == "direction":
                    cs.status = STATUS_DETECTED
                    self._update_direction(cs, pos, float(res.svd_deg))
                elif res.result == "near":
                    cs.status = STATUS_DETECTED
                    self._update_near(cs, pos)
                    clear_res = self._clear(float(pos[0]), float(pos[1]),
                                            channel, phase="SURVEY", cs=cs,
                                            note="near during survey")
                    if clear_res.cleared:
                        cs.status = STATUS_CLEARED
                        self.cleared_count += 1
                    else:
                        # 理论上不会发生；记录并保留状态，后续兜底
                        cs.note_anomaly = True  # type: ignore[attr-defined]
                else:
                    self._update_no_signal(cs, pos)

                self._certify_empty_channels(len(stations))

        self._certify_empty_channels(len(stations))

    # ---------------------------------------------------------------- 阶段B
    def _detected_pending(self) -> List[ChannelState]:
        return [
            cs for cs in self.channels.values()
            if cs.status == STATUS_DETECTED and cs.feasible is not None
        ]

    def _state_center(self, cs: ChannelState) -> np.ndarray:
        if cs.center is not None:
            return np.asarray(cs.center, dtype=float)
        if cs.feasible is not None and len(cs.feasible) > 0:
            return np.mean(cs.feasible, axis=0)
        return np.asarray(self.robot.current_position, dtype=float)

    def _cluster_center(self, cluster: List[ChannelState]) -> np.ndarray:
        centers = [self._state_center(cs) for cs in cluster]
        return np.mean(np.vstack(centers), axis=0)

 
    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray,
                                end: np.ndarray) -> float:
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return float(np.linalg.norm(point - start))
        t = float(np.dot(point - start, segment) / length_sq)
        t = max(0.0, min(1.0, t))
        projection = start + t * segment
        return float(np.linalg.norm(point - projection))

    def _opportunistic_strike(self, target: np.ndarray,
                              exclude_channel: Optional[int] = None) -> None:
        """动态顺路拦截：去目标点前，顺手清除已经足够小的可行域。

        只对 radius <= 20 m 的频道做单点清除，因此不会牺牲“保证清除”的确定性；
        额外路程受 opportunistic_max_detour 限制，避免为了顺手反而绕远。
        """
        if not self.cfg.opportunistic_enabled or not self._time_left():
            return

        start = np.asarray(self.robot.current_position, dtype=float)
        target = np.asarray(target, dtype=float)
        candidates: List[Tuple[float, float, ChannelState]] = []
        for cs in self._detected_pending():
            if cs.channel == exclude_channel or cs.status == STATUS_CLEARED:
                continue
            if cs.radius is None or cs.radius > CLEAR_RADIUS or cs.center is None:
                continue
            center = np.asarray(cs.center, dtype=float)
            detour = (
                math.hypot(center[0] - start[0], center[1] - start[1])
                + math.hypot(target[0] - center[0], target[1] - center[1])
                - math.hypot(target[0] - start[0], target[1] - start[1])
            )
            segment_distance = self._point_segment_distance(center, start, target)
            if detour <= self.cfg.opportunistic_max_detour:
                candidates.append((detour, segment_distance, cs))

        candidates.sort(key=lambda item: (item[0], item[1], item[2].channel))
        for _, _, cs in candidates:
            if not self._time_left() or cs.status == STATUS_CLEARED:
                continue
            if cs.center is None or cs.radius is None or cs.radius > CLEAR_RADIUS:
                continue
            center = np.asarray(cs.center, dtype=float)
            res = self._clear(float(center[0]), float(center[1]), cs.channel,
                              phase="OPPORTUNISTIC", cs=cs,
                              note="opportunistic strike on route")
            self.stats["opportunistic_strikes"] += 1
            if res.cleared:
                self._mark_cleared(cs)
            else:
                self._update_clear_failure(cs, center)

 
    def _phase_tsp_routing(self) -> None:
        """统一目标池 + 动态重规划。

        每轮把所有待处理频道合并成一个任务池：
        - LOCALIZE：可行域仍大的频道，选择下一补测点；
        - CLEAR：半径足够小的频道，直接清除或两次清除；
        - GRID：需要兜底覆盖的频道，每轮只执行一个网格点。

        任务池用带“顺路收益”的贪心 TSP 排序；每执行一个任务后重新建池，
        让新测向、新清除和顺路机会即时进入下一轮决策。
        """
        for cs in self._detected_pending():
            self._recompute_enclosing(cs)

        while self._time_left():
            if self.cleared_count >= self.cfg.stop_at_source_count:
                break
            tasks = self._build_route_tasks()
            if not tasks:
                break
            order = self._tsp_task_order(tasks)
            if not order:
                break
            task = tasks[order[0]]
            self._execute_route_task(task)

    def _build_route_tasks(self) -> List[RouteTask]:
        tasks: List[RouteTask] = []
        robot_pos = np.asarray(self.robot.current_position, dtype=float)

        for cs in self._detected_pending():
            if cs.status == STATUS_CLEARED:
                continue
            if cs.enclosing is None:
                self._recompute_enclosing(cs)

            if cs.radius is not None and cs.radius <= self.cfg.safe_region_radius:
                tasks.append(RouteTask("clear", cs.center.copy(), cs, "safe clear"))
                continue

            if cs.radius is not None and cs.radius > self.cfg.safe_region_radius \
                    and not cs.grid_mode and cs.probe_count < PROBE_MAX_PER_CHANNEL:
                probe = self._peek_probe(cs, robot_pos)
                if probe is not None:
                    point, kind = probe
                    tasks.append(RouteTask("localize", point, cs, kind))
                    continue
                cs.grid_mode = True

            if cs.feasible is not None:
                tasks.append(RouteTask("grid", self._clear_target(cs), cs, "grid step"))

        return tasks

    def _ready_clear_points(self, tasks: List[RouteTask]) -> List[np.ndarray]:
        return [
            np.asarray(task.point, dtype=float)
            for task in tasks
            if task.kind == "clear"
            and task.channel_state.radius is not None
            and task.channel_state.radius <= CLEAR_RADIUS
        ]

    def _edge_cost(self, start: np.ndarray, end: np.ndarray,
                   _ready_clear_points: List[np.ndarray]) -> float:
        return float(np.linalg.norm(end - start))

    def _tsp_task_order(self, tasks: List[RouteTask]) -> List[int]:
        """以欧氏距离为边权的开放 TSP；小任务集用 Held-Karp 精确求解。"""
        if not tasks:
            return []
        if len(tasks) <= TSP_EXACT_LIMIT:
            return self._held_karp_task_order(tasks)

        remaining = set(range(len(tasks)))
        order: List[int] = []
        current = np.asarray(self.robot.current_position, dtype=float)
        ready_clear_points = self._ready_clear_points(tasks)

        while remaining:
            best_idx = min(
                remaining,
                key=lambda idx: (
                    self._edge_cost(current, np.asarray(tasks[idx].point, dtype=float), ready_clear_points),
                    0 if tasks[idx].kind == "clear" else 1 if tasks[idx].kind == "localize" else 2,
                    tasks[idx].channel_state.channel,
                ),
            )
            order.append(best_idx)
            remaining.remove(best_idx)
            current = np.asarray(tasks[best_idx].point, dtype=float)
        points = [task.point for task in tasks]
        return two_opt_open(order, self.robot.current_position, points)

    def _held_karp_task_order(self, tasks: List[RouteTask]) -> List[int]:
        n = len(tasks)
        if n == 0:
            return []
        start = np.asarray(self.robot.current_position, dtype=float)
        ready_clear_points = self._ready_clear_points(tasks)
        points = [np.asarray(task.point, dtype=float) for task in tasks]

        dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}
        for j in range(n):
            mask = 1 << j
            dp[(mask, j)] = (self._edge_cost(start, points[j], ready_clear_points), None)

        for size in range(2, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for item in subset:
                    mask |= 1 << item
                for j in subset:
                    prev_mask = mask ^ (1 << j)
                    best_cost = float("inf")
                    best_prev = None
                    for i in subset:
                        if i == j:
                            continue
                        prev_cost = dp[(prev_mask, i)][0]
                        cost = prev_cost + self._edge_cost(points[i], points[j], ready_clear_points)
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = i
                    dp[(mask, j)] = (best_cost, best_prev)

        full_mask = (1 << n) - 1
        end = min(range(n), key=lambda j: dp[(full_mask, j)][0])
        route: List[int] = []
        mask = full_mask
        current = end
        while current is not None:
            route.append(current)
            _, prev = dp[(mask, current)]
            mask ^= 1 << current
            current = prev
        route.reverse()
        return route

    def _execute_route_task(self, task: RouteTask) -> None:
        cs = task.channel_state
        self._opportunistic_strike(task.point, exclude_channel=cs.channel)
        if cs.status == STATUS_CLEARED:
            return

        if task.kind == "localize":
            self._execute_probe(cs, task.point, task.note)
            self._shared_measure_at(task.point, exclude_channel=cs.channel)
        elif task.kind == "clear":
            self._clear_step(cs)
        else:
            self._grid_clear_step(cs)

    def _max_vertex_distance(self, point: np.ndarray, poly: np.ndarray) -> float:
        if poly is None or len(poly) == 0:
            return float("inf")
        diff = poly - point
        return float(np.max(np.hypot(diff[:, 0], diff[:, 1])))

    def _estimated_intersection_angle_deg(self, point: np.ndarray,
                                          cs: ChannelState) -> float:
        if cs.first_direction is None or cs.center is None:
            return 0.0
        _, old_u = cs.first_direction
        new_vec = np.asarray(cs.center, dtype=float) - point
        norm = float(np.linalg.norm(new_vec))
        if norm < 1e-9:
            return 90.0
        new_u = new_vec / norm
        dot = abs(float(np.dot(old_u, new_u)))
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _shared_measure_at(self, point: np.ndarray,
                           exclude_channel: Optional[int] = None) -> None:
        candidates: List[Tuple[float, ChannelState]] = []
        for cs in self._detected_pending():
            if cs.channel == exclude_channel or cs.status == STATUS_CLEARED:
                continue
            if cs.feasible is None or cs.radius is None:
                continue
            if cs.radius <= self.cfg.safe_region_radius or cs.grid_mode:
                continue
            if self._max_vertex_distance(point, cs.feasible) > 999.0:
                continue
            angle = self._estimated_intersection_angle_deg(point, cs)
            if angle < 20.0:
                continue
            candidates.append((-angle, cs))

        candidates.sort(key=lambda item: (item[0], item[1].channel))
        for _, cs in candidates[:2]:
            if not self._time_left() or cs.status == STATUS_CLEARED:
                break
            res = self._measure(float(point[0]), float(point[1]), cs.channel,
                                phase="SHARED", cs=cs,
                                note="shared measure at same stop")
            self.stats["shared_measures"] += 1
            if res.result == "near":
                self._update_near(cs, point)
                clear_res = self._clear(float(point[0]), float(point[1]), cs.channel,
                                        phase="SHARED", cs=cs,
                                        note="shared near")
                if clear_res.cleared:
                    self._mark_cleared(cs)
            elif res.result == "direction":
                self._update_direction(cs, point, float(res.svd_deg))
            else:
                # 理论上 max vertex <=999 时不会无信号；若模拟器返回了，仍做保守更新。
                self._update_no_signal(cs, point)

    def _build_q2_probe_plan(self, cs: ChannelState,
                             robot_pos: np.ndarray) -> List[np.ndarray]:
        """按问题2思想生成动态拉偏补测点：侧向拉开，形成大交会角。"""
        if cs.first_direction is None:
            return []
        S, u = cs.first_direction
        v = perpendicular_left(u)
        r_est = 600.0
        if cs.center is not None:
            r_est = float(np.linalg.norm(np.asarray(cs.center, dtype=float) - S))
        r_est = float(np.clip(r_est, PROBE_R_EST_MIN, PROBE_R_EST_MAX))
        side_offset = min(r_est * PROBE_SIDE_RATIO, PROBE_SIDE_MAX)
        forward_offset = r_est * PROBE_FORWARD_RATIO
        q2_points = [
            S + forward_offset * u + side_offset * v,
            S + forward_offset * u - side_offset * v,
        ]
        backup_points = [
            S + PROBE_BACKUP_FORWARD * u + PROBE_BACKUP_SIDE * v,
            S + PROBE_BACKUP_FORWARD * u - PROBE_BACKUP_SIDE * v,
        ]
        far_backup_points = [
            S + PROBE_FAR_BACKUP_FORWARD * u + PROBE_FAR_BACKUP_SIDE * v,
            S + PROBE_FAR_BACKUP_FORWARD * u - PROBE_FAR_BACKUP_SIDE * v,
        ]
        q2_points.sort(
            key=lambda point: math.hypot(point[0] - robot_pos[0], point[1] - robot_pos[1])
        )
        backup_points.sort(
            key=lambda point: math.hypot(point[0] - robot_pos[0], point[1] - robot_pos[1])
        )
        far_backup_points.sort(
            key=lambda point: math.hypot(point[0] - robot_pos[0], point[1] - robot_pos[1])
        )
        points = q2_points + backup_points + far_backup_points
        return [np.asarray(point, dtype=float) for point in points]

    def _peek_probe(self, cs: ChannelState,
                    robot_pos: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        if cs.first_direction is None:
            return None

        if not cs.probe_plan and cs.probe_count < 6:
            cs.probe_plan = self._build_q2_probe_plan(cs, robot_pos)

        if cs.probe_plan:
            kind = "q2_candidate" if cs.probe_count < 2 else "q2_backup"
            return np.asarray(cs.probe_plan[0], dtype=float), kind

        if cs.enclosing is None or cs.radius is None or cs.feasible is None:
            return None
        C = cs.center
        R = cs.radius
        e, _ = farthest_pair(cs.feasible)
        n = perpendicular_left(e)
        h = max(40.0, min(250.0, 950.0 - R))
        sign = 1.0 if cs.probe_count % 2 == 0 else -1.0
        point = C + sign * h * n
        if R + h > MIN_RECEIVE_RADIUS:
            point = C.copy()
        if cs.last_probe is not None and \
                math.hypot(point[0] - cs.last_probe[0],
                           point[1] - cs.last_probe[1]) < 1e-6:
            return None
        return np.asarray(point, dtype=float), "adaptive"

    def _next_probe(self, cs: ChannelState,
                    robot_pos: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        """给出该频道下一个补测点；返回 (point, kind) 或 None（应转网格）。"""
        if cs.first_direction is None:
            return None

        # 第一组：问题2的 q2 候选瓣优先；远距离目标再使用接收保证备用点
        if not cs.probe_plan and cs.probe_count < 4:
            cs.probe_plan = self._build_q2_probe_plan(cs, robot_pos)

        if cs.probe_plan:
            point = cs.probe_plan.pop(0)
            if cs.last_probe is not None and \
                    math.hypot(point[0] - cs.last_probe[0],
                               point[1] - cs.last_probe[1]) < 1e-6:
                return None
            kind = "q2_candidate" if cs.probe_count < 2 else "q2_backup"
            return point, kind

        # 自适应横向补测：沿可行域最长轴的垂直方向、在最小包围圆圆心两侧交替
        if cs.enclosing is None or cs.radius is None or cs.feasible is None:
            return None
        C = cs.center
        R = cs.radius
        e, _ = farthest_pair(cs.feasible)
        n = perpendicular_left(e)
        h = max(40.0, min(250.0, 950.0 - R))
        sign = 1.0 if cs.probe_count % 2 == 0 else -1.0
        point = C + sign * h * n
        if R + h > MIN_RECEIVE_RADIUS:
            point = C.copy()
        if cs.last_probe is not None and \
                math.hypot(point[0] - cs.last_probe[0],
                           point[1] - cs.last_probe[1]) < 1e-6:
            return None
        return np.asarray(point, dtype=float), "adaptive"

    def _execute_probe(self, cs: ChannelState, point: np.ndarray, kind: str) -> None:
        R_before = cs.radius if cs.radius is not None else float("inf")
        channel = cs.channel
        res = self._measure(float(point[0]), float(point[1]), channel,
                            phase="LOCALIZE", cs=cs,
                            note=f"probe#{cs.probe_count + 1} {kind}")
        self.stats["probes"] += 1
        cs.probe_count += 1
        cs.last_probe = np.asarray(point, dtype=float)

        if res.result == "near":
            cs.status = STATUS_DETECTED
            self._update_near(cs, point)
            clear_res = self._clear(float(point[0]), float(point[1]), channel,
                                    phase="LOCALIZE", cs=cs, note="near probe")
            if clear_res.cleared:
                cs.status = STATUS_CLEARED
                self.cleared_count += 1
            return

        if res.result == "direction":
            cs.status = STATUS_DETECTED
            self._update_direction(cs, point, float(res.svd_deg))
        else:
            self._update_no_signal(cs, point)

        R_after = cs.radius if cs.radius is not None else float("inf")
        if R_after < R_before - 0.5:
            cs.stagnation = 0
        else:
            cs.stagnation += 1
        if cs.stagnation >= PROBE_STAGNATION_LIMIT:
            cs.grid_mode = True

   
    def _clear_target(self, cs: ChannelState) -> np.ndarray:
        if cs.enclosing is None:
            self._recompute_enclosing(cs)
        if cs.enclosing is None:
            return np.asarray(self.robot.current_position, dtype=float)
        if cs.radius is not None and cs.radius <= self.cfg.safe_region_radius:
            return cs.center.copy()
        if not cs.grid_mode:
            cs.grid_mode = True
            self.stats["grid_channels"] += 1
        grid = getattr(cs, "grid_plan", None)
        if not grid:
            grid = grid_cover_points(cs.feasible, GRID_SPACING)
            cs.grid_plan = self._order_grid(grid)  # type: ignore[attr-defined]
        grid = getattr(cs, "grid_plan", None)
        if grid:
            return np.asarray(grid[0], dtype=float)
        return np.asarray(self.robot.current_position, dtype=float)

    def _clear_step(self, cs: ChannelState) -> bool:
        if cs.status == STATUS_CLEARED:
            return False
        if cs.enclosing is None or cs.radius is None:
            return self._grid_clear_step(cs)

        if cs.radius <= CLEAR_RADIUS:
            if self._single_clear(cs):
                return True
            self._update_clear_failure(cs, cs.center)
            return self._grid_clear_step(cs)

        if cs.radius <= self.cfg.safe_region_radius:
            return self._two_clear(cs)

        return self._grid_clear_step(cs)

    def _order_grid(self, grid: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """蛇形网格按“离机器人较近的一端”起步。"""
        if not grid:
            return []
        start = np.asarray(self.robot.current_position, dtype=float)
        first = np.asarray(grid[0], dtype=float)
        last = np.asarray(grid[-1], dtype=float)
        d_first = math.hypot(first[0] - start[0], first[1] - start[1])
        d_last = math.hypot(last[0] - start[0], last[1] - start[1])
        return grid if d_first <= d_last else grid[::-1]

    def _clear_channel(self, cs: ChannelState) -> bool:
        if cs.status == STATUS_CLEARED:
            return True
        if cs.enclosing is None or cs.radius is None:
            return self._grid_clear(cs)

        if cs.radius <= CLEAR_RADIUS:
            if self._single_clear(cs):
                return True
            self._update_clear_failure(cs, cs.center)
            return self._grid_clear(cs)

        if cs.radius <= self.cfg.safe_region_radius:
            return self._two_clear(cs)

        return self._grid_clear(cs)

    def _mark_cleared(self, cs: ChannelState) -> None:
        if cs.status != STATUS_CLEARED:
            cs.status = STATUS_CLEARED
            self.cleared_count += 1

    def _single_clear(self, cs: ChannelState) -> bool:
        C = cs.center
        res = self._clear(float(C[0]), float(C[1]), cs.channel,
                          phase="CLEAR", cs=cs, note="single clear at center")
        if res.cleared:
            self._mark_cleared(cs)
            return True
        return False

    def _two_clear(self, cs: ChannelState) -> bool:
        """R ≤ 58：先在最小包围圆圆心清除，失败则按测向偏移后再清一次。"""
        C = cs.center.copy()
        R = float(cs.radius)

        first = self._clear(float(C[0]), float(C[1]), cs.channel,
                            phase="CLEAR", cs=cs, note=f"two-clear step1 R={R:.2f}")
        if first.cleared:
            self._mark_cleared(cs)
            return True

        self._update_clear_failure(cs, C)

        m = self._measure(float(C[0]), float(C[1]), cs.channel,
                          phase="CLEAR", cs=cs, note="two-clear bearing at center")
        if m.result == "near":
            res = self._clear(float(C[0]), float(C[1]), cs.channel,
                              phase="CLEAR", cs=cs, note="two-clear near at center")
            if res.cleared:
                self._mark_cleared(cs)
                return True
        elif m.result == "direction":
            u = unit_from_deg(float(m.svd_deg))
            L = (CLEAR_RADIUS + R) / (2.0 * math.cos(math.radians(self.cfg.bearing_error_deg)))
            Q = C + L * u
            res = self._clear(float(Q[0]), float(Q[1]), cs.channel,
                              phase="CLEAR", cs=cs,
                              note=f"two-clear step2 L={L:.2f}")
            if res.cleared:
                self._mark_cleared(cs)
                self.stats["two_clear_used"] += 1
                return True
            self._update_clear_failure(cs, Q)
        # 理论必然成功的第二次清除失败，或测向退化：转网格兜底
        return self._grid_clear(cs)

    def _grid_clear(self, cs: ChannelState) -> bool:
        """20 m 清除圆覆盖剩余可行域，作为确定性兜底。"""
        if cs.feasible is None:
            return False
        grid = getattr(cs, "grid_plan", None)
        if not grid:
            grid = self._order_grid(grid_cover_points(cs.feasible, GRID_SPACING))
            cs.grid_plan = grid  # type: ignore[attr-defined]
        for point in grid:
            if not self._time_left():
                return False
            res = self._clear(float(point[0]), float(point[1]), cs.channel,
                              phase="GRID", cs=cs, note="grid cover")
            self.stats["grid_clears"] += 1
            if res.cleared:
                self._mark_cleared(cs)
                return True
            self._update_clear_failure(cs, np.asarray(point, dtype=float))
        return False

    def _grid_clear_step(self, cs: ChannelState) -> bool:
        """网格兜底的轮转版：每次只清一个网格点，避免单频道长时间霸占。"""
        if cs.feasible is None or not self._time_left():
            return False
        grid = getattr(cs, "grid_plan", None)
        if not grid:
            grid = self._order_grid(grid_cover_points(cs.feasible, GRID_SPACING))
            cs.grid_plan = grid  # type: ignore[attr-defined]
        if not grid:
            return False

        point = grid.pop(0)
        res = self._clear(float(point[0]), float(point[1]), cs.channel,
                          phase="GRID", cs=cs, note="grid cover step")
        self.stats["grid_clears"] += 1
        if res.cleared:
            self._mark_cleared(cs)
            return True
        self._update_clear_failure(cs, np.asarray(point, dtype=float))
        return True

    # ---------------------------------------------------------------- 统计
    def _summary(self) -> Dict[str, object]:
        detected = sum(1 for cs in self.channels.values() if cs.status in (STATUS_DETECTED, STATUS_CLEARED))
        cleared = sum(1 for cs in self.channels.values() if cs.status == STATUS_CLEARED)
        empty = sum(1 for cs in self.channels.values() if cs.status == STATUS_EMPTY)
        unknown = sum(1 for cs in self.channels.values() if cs.status == STATUS_UNKNOWN)

        total_virtual = float(self.robot.virtual_time_s)
        # 理论耗时分解（用于交叉核对）
        action_time = self.stats["measures"] * MEASURE_TIME_S \
            + self.stats["switch_count"] * SWITCH_TIME_S \
            + self.stats["clear_success"] * CLEAR_SUCCESS_TIME_S \
            + (self.stats["clears"] - self.stats["clear_success"]) * CLEAR_FAIL_TIME_S
        move_time = self.stats["move_distance"] / MOVE_SPEED_MPS

        return {
            "channels_total": len(self.channels),
            "detected": detected,
            "cleared": cleared,
            "empty_certified": empty,
            "unknown": unknown,
            "all_resolved": cleared + empty == len(self.channels) and unknown == 0,
            "virtual_time_s": total_virtual,
            "virtual_time_min": total_virtual / 60.0,
            "move_time_s": move_time,
            "action_time_s": action_time,
            "avg_clear_time_s": (total_virtual / cleared) if cleared else None,
            "stats": dict(self.stats),
        }
''')
    # 原始模块：coverage_strategy.py
    _embedded_module('coverage_strategy', r'''"""Independent problem-3 rolling strategies; no HTTP, no access to case truth.

Map certificates use an OUTER domain minus INSCRIBED reception polygons.
Heuristic future coverage is used only in route planning, never in certificates.
"""
from dataclasses import dataclass
import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union, nearest_points

from _q3_model_base import (Problem3Strategy, StrategyConfig, STATUS_UNKNOWN,
    STATUS_DETECTED, STATUS_CLEARED, STATUS_EMPTY, build_survey_points)
from problem3_geometry import (circumscribed_polygon, min_enclosing_circle,
    clip_wedge, clip_halfplane, nearest_neighbor_route, two_opt_open)


def reception(point):
    # Shapely buffers have vertices ON this slightly smaller circle: safe to subtract.
    return Point(float(point[0]), float(point[1])).buffer(999.99, quad_segs=32)


class ChannelCoverage:
    def __init__(self):
        self.domain = Polygon(circumscribed_polygon((0, 0), 1800.00001, n=256))
        self.remaining = {ch:self.domain for ch in range(1,21)}
        self.observations = {ch:[] for ch in range(1,21)}

    def record(self, channel, point, result):
        self.observations[channel].append((tuple(map(float,point)), result))
        # Even positive results establish actual detection coverage. Once detected,
        # that unique channel is handled by the target localisation state machine.
        self.remaining[channel] = self.remaining[channel].difference(reception(point))

    def gain(self, channel, point):
        return self.remaining[channel].intersection(reception(point)).area

    def complete(self, channel):
        return self.remaining[channel].is_empty  # NEVER use small area as certificate


@dataclass
class MapConfig:
    mode: str = 'route_joint_regions_selective_approach'
    scan_gain: float = 350000.0
    max_probes: int = 10
    max_actions: int = 1200
    side: float = 100.0


@dataclass
class Task:
    kind: str
    point: np.ndarray
    channel: int = 0


class CoverageStrategy(Problem3Strategy):
    def __init__(self, robot, config=None, map_config=None):
        super().__init__(robot, config or StrategyConfig(bearing_error_deg=1.01))
        self.mc = map_config or MapConfig()
        self.map = ChannelCoverage()
        self.used = {ch:[] for ch in self.channels}
        self.decisions = []
        self.search_stops = 0
        self.piggyback_measures = 0
        self.piggyback_area = 0.0
        self._grid_cache = {}
        self.pool = [np.array([r*math.cos(a), r*math.sin(a)])
                     for r in (650,1000,1250,1500) for a in np.arange(24)*math.pi/12]
        self.pool_disks = [reception(p) for p in self.pool]

    def _measure(self, x, y, channel, phase, cs=None, note=''):
        self._check_action_budget(x,y,6)
        res = super()._measure(x,y,channel,phase,cs,note)
        self.map.record(channel,(x,y),res.result)
        self.used[channel].append(np.array([x,y]))
        return res

    def _clear(self,x,y,channel,phase,cs=None,note=''):
        self._check_action_budget(x,y,5)
        return super()._clear(x,y,channel,phase,cs,note)

    def _check_action_budget(self,x,y,action_s):
        if not self._time_left():
            raise RuntimeError('Insufficient real or virtual time')
        limit=getattr(self.robot,'max_virtual_duration_s',360000)
        if limit is not None and self.robot.virtual_time_s+self.robot.estimate_move_time(x,y)+action_s>=limit:
            raise RuntimeError('Next action exceeds virtual-time budget')

    def _recompute_enclosing(self, cs):
        # Do not discard a tiny but legitimate set, and independently inflate the
        # circle to contain EVERY polygon vertex even if MEC has roundoff.
        if cs.feasible is None or len(cs.feasible)==0:
            raise RuntimeError('Empty feasible region; completion cannot be certified')
        c,r=min_enclosing_circle(cs.feasible)
        r=max(r,float(np.max(np.linalg.norm(cs.feasible-c,axis=1))))+1e-6
        cs.enclosing=(c,r)

    def certify(self):
        upper = self._known_source_count()==16
        for ch,cs in self.channels.items():
            if cs.status==STATUS_UNKNOWN and (upper or self.map.complete(ch)):
                self._mark_empty(cs,'16 known sources' if upper else 'actual per-channel coverage complete')

    def unknown(self):
        return [ch for ch,s in self.channels.items() if s.status==STATUS_UNKNOWN]

    def sense(self,ch,p,phase):
        cs=self.channels[ch]
        res=self._measure(*map(float,p),ch,phase,cs)
        if res.result=='direction':
            cs.status=STATUS_DETECTED
            self._update_direction(cs,p,float(res.svd_deg))
        elif res.result=='near':
            cs.status=STATUS_DETECTED
            self._update_near(cs,p)
            if not self._single_clear(cs):
                raise RuntimeError('near clear failed')
        elif res.result=='no_signal':
            self._update_no_signal(cs,p)
        else:
            raise RuntimeError('Invalid measurement result')
        self.certify()
        return res

    def scan(self,p,force=False):
        current=self.robot.current_channel
        order=sorted(self.unknown(),key=lambda ch:(ch!=current,ch))
        changed=False
        for ch in order:
            if self.channels[ch].status!=STATUS_UNKNOWN:
                continue
            if not self._time_left():
                raise RuntimeError('Time exhausted during scan')
            gain=self.map.gain(ch,p)
            threshold=1000000.0 if 'selective' in self.mc.mode else self.mc.scan_gain
            if gain<=0 or (not force and gain<threshold):
                continue
            self.sense(ch,p,'SEARCH' if force else 'PIGGYBACK')
            if not force:
                self.piggyback_measures+=1
                self.piggyback_area+=gain
            changed=True
        if changed and force:
            self.search_stops+=1
        return changed

    def remaining_shape(self):
        return unary_union([self.map.remaining[ch] for ch in self.unknown()])

    def search_point(self,need):
        if need.is_empty:
            return None
        pos=np.array(self.robot.current_position)
        gains=np.array([need.intersection(d).area for d in self.pool_disks])
        best=float(gains.max())
        if best>1e-8:
            ids=np.flatnonzero(gains>=(0.99 if 'joint' in self.mc.mode else 0.65)*best)
            idx=min(ids,key=lambda j:float(np.linalg.norm(self.pool[j]-pos)))
            return self.pool[int(idx)].copy()
        # Handles arbitrarily small slivers without falsely declaring completion.
        p=need.representative_point()
        return np.array([p.x,p.y])

    def local_point(self,cs):
        pos=np.array(self.robot.current_position)
        if self.mc.mode.endswith('approach') or self.mc.mode in ('nn_approach','hex_online'):
            # Move toward the estimated target BEFORE seeking lateral parallax.
            # A small sideways component is cheaper than an early large detour.
            c=cs.center
            vec=c-pos; d=np.linalg.norm(vec)
            u=vec/d if d>1e-6 else cs.first_direction[1]
            n=np.array([-u[1],u[0]])
            h=min(self.mc.side,max(30,cs.radius*0.2))
            candidates=[c+h*n,c-h*n]
            if cs.probe_count>=3:
                candidates+=self._build_q2_probe_plan(cs,pos)
        else:
            candidates=self._build_q2_probe_plan(cs,pos)
        candidates=[p for p in candidates if all(np.linalg.norm(p-q)>1 for q in self.used[cs.channel])]
        if not candidates:
            return None
        return min(candidates,key=lambda p:float(np.linalg.norm(p-pos)))

    def target_tasks(self):
        tasks=[]
        for ch,cs in self.channels.items():
            if cs.status!=STATUS_DETECTED:
                continue
            if cs.radius is None:
                raise RuntimeError('Detected source has no enclosing region')
            if cs.radius<=self.cfg.safe_region_radius:
                tasks.append(Task('clear',cs.center.copy(),ch))
            elif cs.probe_count<self.mc.max_probes:
                p=self.local_point(cs)
                tasks.append(Task('probe',p,ch) if p is not None else Task('grid',cs.center.copy(),ch))
            else:
                tasks.append(Task('grid',cs.center.copy(),ch))
        return tasks

    def planned_searches(self,tasks):
        need=self.remaining_shape()
        # This is a planning estimate, NOT a map update or certificate. All
        # assumptions about future scans are discarded after the next real action.
        if self.mc.mode.startswith('route'):
            for t in tasks:
                need=need.difference(reception(t.point))
                if t.kind=='probe':
                    need=need.difference(reception(self.channels[t.channel].center))
        searches=[]
        for _ in range(12):
            if need.is_empty:
                break
            p=self.region_search_point(need,tasks+searches) if 'regions' in self.mc.mode else self.search_point(need)
            searches.append(Task('search',p))
            updated=need.difference(reception(p))
            if updated.equals(need):
                raise RuntimeError('No coverage progress')
            need=updated
        return searches

    def region_search_point(self,need,tasks):
        """Cover a whole residual component from a region of possible scan sites,
        then choose a site close to the currently planned travel path.
        This alters the coverage task itself, not just its visiting order.
        """
        pieces=list(need.geoms) if hasattr(need,'geoms') else [need]
        piece=max(pieces,key=lambda g:g.area)
        if 'batch' in self.mc.mode:
            # Several disconnected blind patches may share one reception disk.
            # Merge feasible groups before choosing a detection region near route.
            for other in sorted(pieces,key=lambda g:g.distance(piece)):
                if other.equals(piece):
                    continue
                combined=piece.union(other)
                h=combined.convex_hull
                if h.geom_type!='Polygon':
                    continue
                vv=np.array(h.exterior.coords)[:-1]
                cc,rr=min_enclosing_circle(vv)
                rr=max(rr,float(np.max(np.linalg.norm(vv-cc,axis=1))))
                if rr<995:
                    piece=combined
        hull=piece.convex_hull
        if hull.geom_type!='Polygon':
            return self.search_point(need)
        vertices=np.array(hull.exterior.coords)[:-1]
        c,r=min_enclosing_circle(vertices)
        r=max(r,float(np.max(np.linalg.norm(vertices-c,axis=1))))+0.001
        # Inscribed reception polygon has inradius ~999.689m.
        if r<995:
            allowed=Point(*c).buffer(999.5-r,quad_segs=24)
            pts=[np.asarray(self.robot.current_position)]+[t.point for t in tasks]
            if len(pts)>2:
                order=nearest_neighbor_route(pts[0],pts[1:])
                order=two_opt_open(order,pts[0],pts[1:])
                pts=[pts[0]]+[pts[j+1] for j in order]
            path=LineString(pts) if len(pts)>1 else Point(*pts[0])
            q=nearest_points(allowed,path)[0]
            return np.array([q.x,q.y])
        return self.search_point(need)

    def choose(self):
        tasks=self.target_tasks()
        if self.mc.mode.startswith('route'):
            tasks+=self.planned_searches(tasks)
            if not tasks:
                return None
            points=[t.point for t in tasks]
            route=nearest_neighbor_route(self.robot.current_position,points)
            route=two_opt_open(route,self.robot.current_position,points)
            return tasks[route[0]]
        p=self.search_point(self.remaining_shape()) if self.unknown() else None
        if p is not None:
            tasks.append(Task('search',p))
        if not tasks:
            return None
        pos=np.array(self.robot.current_position)
        def cost(t):
            action=5*len(self.unknown())+max(0,len(self.unknown())-1) if t.kind=='search' else 6
            return np.linalg.norm(t.point-pos)/5+action
        return min(tasks,key=cost)

    def execute(self,t):
        self.decisions.append(dict(kind=t.kind,point=t.point.tolist(),channel=t.channel,
                                   time_s=self.robot.virtual_time_s,unknown=len(self.unknown())))
        if t.kind=='search':
            if not self.scan(t.point,force=True):
                raise RuntimeError('Search action made no progress')
        elif t.kind=='probe':
            cs=self.channels[t.channel]
            cs.probe_count+=1; self.stats['probes']+=1
            self.sense(t.channel,t.point,'LOCALIZE')
        elif t.kind=='clear':
            if not self._clear_channel(self.channels[t.channel]):
                raise RuntimeError('Clear failed')
        else:
            # Finite plan is generated once. Parent's regenerating exhausted grid
            # is not used: failure here is explicit instead of an infinite loop.
            if not self._grid_clear(self.channels[t.channel]):
                raise RuntimeError('Finite grid exhausted')
        if self.mc.mode not in ('nn_separate',):
            self.scan(np.array(self.robot.current_position),force=False)
        if 'joint' in self.mc.mode:
            self.shared_localize(np.array(self.robot.current_position))
        self.certify()

    def shared_localize(self,p):
        # A stop can triangulate several already detected channels. This avoids
        # repeatedly returning to each source's original (stale) first-probe area.
        for ch,cs in self.channels.items():
            if cs.status!=STATUS_DETECTED or cs.radius is None or cs.radius<=58:
                continue
            if any(np.linalg.norm(p-q)<60 for q in self.used[ch]):
                continue
            if np.linalg.norm(p-cs.center)>1200:
                continue
            if self._estimated_intersection_angle_deg(p,cs)<12:
                continue
            self.sense(ch,p,'SHARED_LOCALIZE')
            self.stats['shared_measures']+=1

    def run_hex(self):
        # Reconstructed descriptive baseline: full known/unknown scan, then service
        # all discovered targets before the next station. Rotate ring toward first
        # post-origin destination. It is NOT claimed to be the user's missing file.
        ring=build_survey_points()[1:]
        self.scan(np.array([0.,0.]),True)
        idx=0
        while True:
            tasks=self.target_tasks()
            if tasks:
                p=np.array(self.robot.current_position)
                t=min(tasks,key=lambda t:np.linalg.norm(t.point-p))
                self.execute(t)
            elif idx<len(ring) and self.unknown():
                p=np.array(self.robot.current_position)
                j=min(range(idx,len(ring)),key=lambda k:np.linalg.norm(ring[k]-p))
                ring[idx],ring[j]=ring[j],ring[idx]
                self.scan(ring[idx],True); idx+=1
            else:
                break

    def run(self):
        self._enter()
        try:
            if self.mc.mode=='hex_online':
                self.run_hex()
            else:
                self.scan(np.array([0.,0.]),True)
                for _ in range(self.mc.max_actions):
                    if not self._time_left():
                        raise RuntimeError('Time limit; no completion certificate')
                    t=self.choose()
                    if t is None:
                        break
                    self.execute(t)
                else:
                    raise RuntimeError('Action budget exhausted')
            self.certify()
            result=self._summary()
            if not result['all_resolved'] or not 10<=result['cleared']<=16:
                raise RuntimeError('Incomplete search or uncleared target')
            result.update(search_stops=self.search_stops,piggyback_measures=self.piggyback_measures,
                          piggyback_area_m2=self.piggyback_area,policy=self.mc.mode)
            return result
        finally:
            self._exit()
''')
    # 原始模块：lookahead_strategy.py
    _embedded_module('lookahead_strategy', r'''"""Q3 experiments: observation rollout and channel-aware scan value.

Hypothetical observations are local arrays only. Certificates remain the frozen
CoverageStrategy's actual-observation maps and conservative feasible polygons.
"""
from dataclasses import dataclass
import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union, nearest_points

from coverage_strategy import CoverageStrategy, MapConfig, Task, reception
from _q3_model_base import STATUS_DETECTED, STATUS_UNKNOWN
from problem3_geometry import clip_wedge, clip_halfplane, min_enclosing_circle, nearest_neighbor_route, two_opt_open

BASE = 'route_joint_regions_selective_approach'


@dataclass
class LookaheadConfig:
    probes: bool = True
    scan_value: bool = False
    depth: int = 2
    history: bool = False
    joint_order: bool = False
    full_regions: bool = False


class LookaheadStrategy(CoverageStrategy):
    def __init__(self, robot, config=None, options=None):
        super().__init__(robot, config, MapConfig(mode=BASE))
        self.options = options or LookaheadConfig()
        self._particles_cache = {}
        self.lookahead_stats = dict(probe_candidates=0, value_scans=0, value_skips=0)

    def _recompute_enclosing(self, cs):
        if self.options.history and cs.feasible is not None and len(cs.feasible)>0:
            observations = self.map.observations[cs.channel]
            misses = [np.array(p) for p,result in observations if result=='no_signal']
            hits = [np.array(p) for p,result in observations if result in ('direction','near')]
            # The unknown reception radius is fixed for this channel. Every
            # no-signal point must be farther from G than every signal point.
            for q in misses:
                for p in hits:
                    v=q-p; rhs=float(np.dot(q,q)-np.dot(p,p))
                    cs.feasible=clip_halfplane(cs.feasible,-2*v[0],-2*v[1],rhs)
            if misses and len(cs.feasible)>=3:
                remaining=Polygon(cs.feasible).difference(unary_union([reception(q) for q in misses]))
                if remaining.is_empty:
                    raise RuntimeError('Actual observation history has empty feasible region')
                hull=remaining.convex_hull
                if hull.geom_type=='Polygon':
                    # Taking the convex hull enlarges the valid nonconvex set.
                    cs.feasible=np.asarray(hull.exterior.coords)[:-1].copy()
        super()._recompute_enclosing(cs)

    def region_search_point(self, need, tasks):
        if not self.options.full_regions:
            return super().region_search_point(need,tasks)
        pieces=list(need.geoms) if hasattr(need,'geoms') else [need]
        hull=max(pieces,key=lambda g:g.area).convex_hull
        if hull.geom_type!='Polygon':
            return super().region_search_point(need,tasks)
        vertices=np.asarray(hull.exterior.coords)[:-1]
        c,r=min_enclosing_circle(vertices)
        if r>=995:
            return super().region_search_point(need,tasks)
        # Intersect the permissible scan disks of EVERY hull vertex. This uses
        # more of the feasible scan-site region than B(c,999.5-r). Each disk is
        # inscribed, radius < actual coverage polygon's inradius.
        allowed=None
        for v in vertices:
            disk=Point(*v).buffer(999.5,quad_segs=32)
            allowed=disk if allowed is None else allowed.intersection(disk)
        if allowed.is_empty:
            return super().region_search_point(need,tasks)
        points=[np.asarray(self.robot.current_position)]+[t.point for t in tasks]
        if len(points)>2:
            order=two_opt_open(nearest_neighbor_route(points[0],points[1:]),points[0],points[1:])
            points=[points[0]]+[points[j+1] for j in order]
        path=LineString(points) if len(points)>1 else Point(*points[0])
        p=nearest_points(allowed,path)[0]
        return np.array([p.x,p.y])

    def particles(self, cs):
        """Deterministic area-stratified convex-polygon samples; not simulator truth.
        Uniform position and conditional uniform reception radius are planning assumptions.
        """
        key = (cs.channel, cs.feasible.tobytes(), len(self.map.observations[cs.channel]))
        if key in self._particles_cache:
            return self._particles_cache[key]
        poly = cs.feasible
        center = poly.mean(axis=0)
        edge = np.roll(poly, -1, axis=0)
        areas = np.abs((poly[:,0]-center[0])*(edge[:,1]-center[1])
                       -(poly[:,1]-center[1])*(edge[:,0]-center[0]))
        cum = np.cumsum(areas)
        samples = []
        for i in range(7):
            j = min(len(poly)-1, int(np.searchsorted(cum, (i+.5)/7*cum[-1])))
            a = math.sqrt(((i*.61803398875+.31)%1)*.8+.1)
            b = (i*.41421356237+.23)%1
            g = (1-a)*center+a*((1-b)*poly[j]+b*edge[j])
            low, high = 1000., 1500.
            for p, result in self.map.observations[cs.channel]:
                d = float(np.linalg.norm(g-p))
                if result in ('direction', 'near'):
                    low = max(low,d)
                elif result == 'no_signal':
                    high = min(high,d)
            if low <= high:
                samples.append((g, (low+high)/2))
        if not samples:
            # Uninformative planning fallback, never changes actual feasibility.
            samples = [(cs.center.copy(), 1500.)]
        self._particles_cache[key] = samples
        return samples

    def candidates(self, cs, origin, successor=None):
        c = cs.center
        v = c-origin
        d = np.linalg.norm(v)
        u = v/d if d>1e-6 else np.array([1.,0.])
        n = np.array([-u[1],u[0]])
        h = min(100.,max(30.,cs.radius*.2))
        points = [super().local_point(cs), origin.copy()]
        for fraction in (.35,.7,1.):
            for side in (-1,1):
                points.append(origin+fraction*v+side*h*n)
        if successor is not None:
            points.append((c+successor)*.5)
        # Already-planned stops can supply useful bearings to multiple sources.
        others = [s.center for s in self.channels.values()
                  if s.status==STATUS_DETECTED and s.channel!=cs.channel]
        points += sorted(others,key=lambda p:np.linalg.norm(p-origin))[:2]
        out = []
        for p in points:
            if p is None or any(np.linalg.norm(p-q)<2 for q in self.used[cs.channel]):
                continue
            if not any(np.linalg.norm(p-q)<2 for q in out):
                out.append(np.asarray(p).copy())
        return out

    def finish_branch(self, poly, p, g, receive, error, depth, first_signal):
        """Hypothetical measure -> optional second measure -> clear sequence.
        Returns cost after arrival at p and the predicted finishing position.
        Geometry calculations affect no live ChannelState.
        """
        distance = float(np.linalg.norm(g-p))
        if distance<=5:
            return 11., p
        if distance>receive:
            v = p-first_signal
            rhs = float(np.dot(p,p)-np.dot(first_signal,first_signal))
            posterior = clip_halfplane(poly,-2*v[0],-2*v[1],rhs)
        else:
            angle = math.atan2(g[1]-p[1],g[0]-p[0])+math.radians(error)
            posterior = clip_wedge(poly,p,angle,math.radians(self.cfg.bearing_error_deg))
        if len(posterior)==0:
            return 1000., p
        c,r = min_enclosing_circle(posterior)
        travel = float(np.linalg.norm(c-p))/5
        if r<=58:
            if np.linalg.norm(c-g)<=20:
                return 6+travel+5,c
            length = (20+r)/(2*math.cos(math.radians(self.cfg.bearing_error_deg)))
            a = math.atan2(g[1]-c[1],g[0]-c[0])+math.radians(error)
            end = c+length*np.array([math.cos(a),math.sin(a)])
            return 6+travel+3+6+length/5+5,end
        if depth>1:
            v = c-p
            d = np.linalg.norm(v)
            u = v/d if d>1e-6 else np.array([1.,0.])
            n = np.array([-u[1],u[0]])
            h = min(100.,max(30.,r*.2))
            q = min((c+h*n,c-h*n),key=lambda q:np.linalg.norm(q-p))
            cost,end = self.finish_branch(posterior,q,g,receive,error,depth-1,first_signal)
            return 6+float(np.linalg.norm(q-p))/5+cost,end
        # Finite-horizon continuation estimate; it is not a completion certificate.
        return 6+travel+12+2*r/5,c

    def probe_score(self, cs, p, before, after):
        values = []
        for g, receive in self.particles(cs):
            for error in (-.9,.9):
                cost,end = self.finish_branch(cs.feasible,p,g,receive,error,
                                              self.options.depth,cs.first_direction[0])
                if after is not None:
                    cost += float(np.linalg.norm(end-after))/5
                values.append(cost)
        return float(np.linalg.norm(p-before))/5+float(np.mean(values))

    def choose(self):
        if not self.options.probes:
            return super().choose()
        tasks = self.target_tasks()
        tasks += self.planned_searches(tasks)
        if not tasks:
            return None
        points = [t.point for t in tasks]
        order = two_opt_open(nearest_neighbor_route(self.robot.current_position,points),
                             self.robot.current_position,points)
        # Optimize the near-term route's actual first probe; other task positions
        # remain continuation estimates and will be recomputed after feedback.
        first = tasks[order[0]]
        if self.options.joint_order:
            before=np.asarray(self.robot.current_position)
            best=None
            best_score=float('inf')
            for index in order[:3]:
                t=tasks[index]
                remaining=[tasks[j].point for j in order if j!=index]
                if t.kind=='probe':
                    cs=self.channels[t.channel]
                    candidates=self.candidates(cs,before,remaining[0] if remaining else None)
                else:
                    candidates=[t.point]
                for p in candidates:
                    if t.kind=='probe':
                        ends=[]; costs=[]
                        for g,receive in self.particles(cs):
                            cost,end=self.finish_branch(cs.feasible,p,g,receive,0.,self.options.depth,cs.first_direction[0])
                            ends.append(end); costs.append(cost)
                        end=np.mean(ends,axis=0)
                        score=np.linalg.norm(p-before)/5+np.mean(costs)
                    else:
                        end=p
                        score=np.linalg.norm(p-before)/5+(6*len(self.unknown()) if t.kind=='search' else 6)
                    if remaining:
                        route=two_opt_open(nearest_neighbor_route(end,remaining),end,remaining)
                        last=end
                        for j in route:
                            score+=np.linalg.norm(remaining[j]-last)/5
                            last=remaining[j]
                    # Common future action costs cancel; compensate for which
                    # task was explicitly completed in the finite rollout.
                    score-=6*len(self.unknown()) if t.kind=='search' else 6
                    if score<best_score:
                        best_score=score; best=Task(t.kind,p.copy(),t.channel)
            if best is not None:
                return best
        if first.kind=='probe':
            cs = self.channels[first.channel]
            before = np.asarray(self.robot.current_position)
            after = tasks[order[1]].point if len(order)>1 else None
            candidates = self.candidates(cs,before,after)
            if candidates:
                self.lookahead_stats['probe_candidates'] += len(candidates)
                first.point = min(candidates,key=lambda p:self.probe_score(cs,p,before,after))
        return first

    def scan_plan_cost(self, remaining, points, start):
        """Greedy per-channel scan allocation plus explicit residual coverage travel.
        All predicted disks stay inside this local planning copy.
        """
        need = dict(remaining)
        cost = 0.
        disks = [reception(p) for p in points]
        for i,disk in enumerate(disks):
            later = unary_union(disks[i+1:]) if i+1<len(disks) else None
            for ch,shape in list(need.items()):
                contribution = shape.intersection(disk)
                if contribution.is_empty:
                    continue
                # Defer fully redundant work to a later planned stop.
                if later is not None and contribution.difference(later).is_empty:
                    continue
                need[ch] = shape.difference(disk)
                cost += 6
        pos = np.asarray(points[-1] if points else start)
        for _ in range(12):
            union = unary_union(list(need.values()))
            if union.is_empty:
                break
            p = self.region_search_point(union,[Task('anchor',pos)])
            disk = reception(p)
            cost += float(np.linalg.norm(p-pos))/5
            for ch,shape in list(need.items()):
                if not shape.intersection(disk).is_empty:
                    need[ch] = shape.difference(disk)
                    cost += 6
            pos = p
        return cost

    def scan(self,p,force=False):
        if force or not self.options.scan_value:
            return super().scan(p,force)
        unknown = self.unknown()
        if not unknown:
            return False
        # Group candidates: their saved return trip must not be counted once per channel.
        disk = reception(p)
        eligible = [ch for ch in unknown if not self.map.remaining[ch].intersection(disk).is_empty]
        if not eligible:
            return False
        targets = self.target_tasks()
        points = [t.point for t in targets if np.linalg.norm(t.point-p)>30]
        if points:
            order = two_opt_open(nearest_neighbor_route(p,points),p,points)
            points = [points[j] for j in order]
        remaining = {ch:self.map.remaining[ch] for ch in unknown}
        without = self.scan_plan_cost(remaining,points,p)
        groups = [eligible, [ch for ch in eligible if self.map.gain(ch,p)>=1000000],
                  [ch for ch in eligible if remaining[ch].difference(disk).is_empty]]
        best_group = []
        best_cost = without
        for group in groups:
            if not group:
                continue
            hypothetical = {ch:(shape.difference(disk) if ch in group else shape)
                            for ch,shape in remaining.items()}
            cost = 6*len(group)+self.scan_plan_cost(hypothetical,points,p)
            if cost < best_cost-1:
                best_cost,best_group = cost,group
        self.lookahead_stats['value_skips'] += len(eligible)-len(best_group)
        changed = False
        for ch in sorted(best_group,key=lambda ch:(ch!=self.robot.current_channel,ch)):
            if self.channels[ch].status!=STATUS_UNKNOWN:
                continue
            gain = self.map.gain(ch,p)
            self.sense(ch,p,'VALUE_SCAN')
            self.piggyback_measures += 1
            self.piggyback_area += gain
            self.lookahead_stats['value_scans'] += 1
            changed = True
        return changed

    def run(self):
        result = super().run()
        result.update(lookahead=self.lookahead_stats)
        return result
''')
    from client import RobotClient
    from lookahead_strategy import LookaheadStrategy as CoverageStrategy, LookaheadConfig as MapConfig
    from _q3_model_base import StrategyConfig
    return RobotClient, CoverageStrategy, MapConfig, StrategyConfig


# =========================================================================== #
# 策略层（原 q3_empty_channel_strategy.py）
#
# 统一入口：
#   * build_empty_channel_strategy —— 低源"空频道"选择性顺带覆盖（默认策略）；
#   * build_hex_cover_strategy     —— 在上一者基础上加认证六边形覆盖骨架；
#   * build_strategy(...)          —— 按开关组装并连同内嵌模型一起返回。
# 默认（不带任何开关）即最终演练采用的组合。
# =========================================================================== #

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# 本地随机测试所需的题面常量与 LocalSimulator 均由根目录 simulator.py 提供
# （已在模块顶部导入；simulator.py 不反向依赖本模块，无循环导入）。

_HERE = Path(__file__).resolve().parent

DEFAULT_PIGGYBACK_GAIN_M2 = 1_000_000.0
DEFAULT_HEX_RING_RADIUS_M = 1_125.0
DISABLED_PIGGYBACK_GAIN_M2 = 1.0e30


def build_empty_channel_strategy(
    base_strategy,
    gain_threshold_m2: float,
    always_finish_channel: bool = True,
    resume_piggyback_at_known_sources: int | None = None,
    resumed_gain_threshold_m2: float = DEFAULT_PIGGYBACK_GAIN_M2,
    resume_by_search_stop: int | None = None,
):
    from coverage_strategy import reception

    class EmptyChannelStrategy(base_strategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._adaptive_resume_latched = False
            self._adaptive_resume_known_sources = None
            self._adaptive_resume_search_stops = None

        def _adaptive_resume_active(self):
            if self._adaptive_resume_latched:
                return True
            if resume_piggyback_at_known_sources is None:
                return False
            known_sources = self._known_source_count()
            if known_sources < resume_piggyback_at_known_sources:
                return False
            if resume_by_search_stop is not None and self.search_stops > resume_by_search_stop:
                return False
            self._adaptive_resume_latched = True
            self._adaptive_resume_known_sources = known_sources
            self._adaptive_resume_search_stops = self.search_stops
            return True

        def scan(self, point, force=False):
            if force:
                return super().scan(point, force=True)

            unknown = self.unknown()
            if not unknown:
                return False
            disk = reception(point)
            current = self.robot.current_channel
            resume_active = self._adaptive_resume_active()
            active_threshold = (
                resumed_gain_threshold_m2 if resume_active else gain_threshold_m2
            )
            candidates = []
            for channel in unknown:
                if self.channels[channel].status != "UNKNOWN":
                    continue
                remaining = self.map.remaining[channel]
                gain = remaining.intersection(disk).area
                completes = not remaining.is_empty and remaining.difference(disk).is_empty
                if (always_finish_channel and completes) or gain >= active_threshold:
                    candidates.append((channel != current, channel, gain, completes))

            changed = False
            for _, channel, gain, completes in sorted(candidates):
                if self.channels[channel].status != "UNKNOWN":
                    continue
                self.sense(channel, point, "PIGGYBACK")
                self.piggyback_measures += 1
                self.piggyback_area += gain
                changed = True
            return changed

        def run(self):
            result = super().run()
            result["adaptive_resume"] = {
                "latched": self._adaptive_resume_latched,
                "known_sources": self._adaptive_resume_known_sources,
                "search_stops": self._adaptive_resume_search_stops,
            }
            return result

    return EmptyChannelStrategy


def build_hex_cover_strategy(
    base_strategy,
    gain_threshold_m2: float,
    always_finish_channel: bool = False,
    ring_radius_m: float = DEFAULT_HEX_RING_RADIUS_M,
    resume_piggyback_at_known_sources: int | None = None,
    resumed_gain_threshold_m2: float = DEFAULT_PIGGYBACK_GAIN_M2,
    resume_dynamic_search: bool = False,
    resume_by_search_stop: int | None = None,
    enable_marginal_scan: bool = False,
    marginal_scan_margin_s: float = 2.0,
    enable_neighborhood_cover: bool = False,
    neighborhood_iterations: int = 4,
):
    """Add a certified center-plus-six coverage backbone.

    The center is already force-scanned by the parent strategy.  Six sites on
    the regular hexagon cover the full search domain.  At each replanning step
    all 64 subsets are checked against the *actual* per-channel residual maps;
    this can omit a site already made redundant by real measurements.  If a
    numerical or geometric residual remains after the backbone is exhausted,
    the parent's certified residual planner is used unchanged.
    """
    from coverage_strategy import Task, reception
    from problem3_geometry import nearest_neighbor_route, route_length, two_opt_open
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points, unary_union

    threshold_strategy = build_empty_channel_strategy(
        base_strategy,
        gain_threshold_m2,
        always_finish_channel=always_finish_channel,
        resume_piggyback_at_known_sources=resume_piggyback_at_known_sources,
        resumed_gain_threshold_m2=resumed_gain_threshold_m2,
        resume_by_search_stop=resume_by_search_stop,
    )

    class HexCoverStrategy(threshold_strategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._hex_points = None
            self._hex_disks = None
            self._hex_orientation_rad = None
            self._marginal_scan_evaluations = 0
            self._marginal_scan_accepts = 0
            self._marginal_scan_planned_saving_s = 0.0
            self._cover_patches = None
            self._cover_allowed = None
            self._cover_neighborhood_points = None
            self._cover_neighborhood_updates = 0
            self._cover_neighborhood_planned_saving_m = 0.0

        def _route_distance(self, points):
            if not points:
                return 0.0
            start = self.robot.current_position
            order = nearest_neighbor_route(start, points)
            order = two_opt_open(order, start, points)
            return route_length(order, start, points)

        def _initialize_hex_cover(self, tasks):
            if self._hex_points is not None:
                return
            sector = math.pi / 3.0
            candidate_angles = [k * math.pi / 36.0 for k in range(12)]
            candidate_angles.extend(
                math.atan2(float(task.point[1]), float(task.point[0])) % sector
                for task in tasks
            )
            task_points = [np.asarray(task.point, dtype=float) for task in tasks]
            best = None
            for angle in candidate_angles:
                ring = [
                    np.array(
                        [
                            ring_radius_m * math.cos(angle + k * sector),
                            ring_radius_m * math.sin(angle + k * sector),
                        ]
                    )
                    for k in range(6)
                ]
                cover = unary_union([reception((0.0, 0.0))] + [reception(p) for p in ring])
                if not self.map.domain.difference(cover).is_empty:
                    continue
                score = self._route_distance(task_points + ring)
                key = (score, angle)
                if best is None or key < best[0]:
                    best = (key, angle, ring)
            if best is None:
                raise RuntimeError("The requested hexagon radius does not certify full coverage")
            _, self._hex_orientation_rad, self._hex_points = best
            self._hex_disks = [reception(point) for point in self._hex_points]
            if enable_neighborhood_cover:
                self._initialize_cover_neighborhoods()

        def _initialize_cover_neighborhoods(self):
            safe_radius_m = 999.5
            far_radius_m = 4_000.0
            sector = math.pi / 3.0
            center_disk = reception((0.0, 0.0))
            self._cover_patches = []
            self._cover_allowed = []
            self._cover_neighborhood_points = []
            for index, fixed_point in enumerate(self._hex_points):
                angle = self._hex_orientation_rad + index * sector
                wedge = Polygon(
                    [
                        (0.0, 0.0),
                        (
                            far_radius_m * math.cos(angle - sector / 2.0),
                            far_radius_m * math.sin(angle - sector / 2.0),
                        ),
                        (
                            far_radius_m * math.cos(angle + sector / 2.0),
                            far_radius_m * math.sin(angle + sector / 2.0),
                        ),
                    ]
                )
                patch = self.map.domain.intersection(wedge).difference(center_disk).convex_hull
                vertices = list(patch.exterior.coords)[:-1]
                allowed = Point(vertices[0]).buffer(safe_radius_m, quad_segs=32)
                for vertex in vertices[1:]:
                    allowed = allowed.intersection(
                        Point(vertex).buffer(safe_radius_m, quad_segs=32)
                    )
                if allowed.is_empty:
                    raise RuntimeError("A certified outer-sector scan neighborhood is empty")
                if not patch.difference(reception(fixed_point)).is_empty:
                    raise RuntimeError("The fixed hexagon point does not cover its assigned sector")
                self._cover_patches.append(patch)
                self._cover_allowed.append(allowed)
                self._cover_neighborhood_points.append(np.asarray(fixed_point, dtype=float))

        @staticmethod
        def _point_from_geometry(geometry):
            return np.array([float(geometry.x), float(geometry.y)])

        def _optimized_neighborhood_searches(self, tasks):
            remaining = {channel: self.map.remaining[channel] for channel in self.unknown()}
            required = [
                index
                for index, patch in enumerate(self._cover_patches)
                if any(shape.intersection(patch).area > 1e-8 for shape in remaining.values())
            ]
            if not required:
                return []

            fixed_task_points = [np.asarray(task.point, dtype=float) for task in tasks]
            points = {
                index: np.asarray(self._cover_neighborhood_points[index], dtype=float).copy()
                for index in required
            }
            original_points = {
                index: np.asarray(self._hex_points[index], dtype=float).copy()
                for index in required
            }
            fixed_route_points = fixed_task_points + [original_points[index] for index in required]
            fixed_route_m = self._route_distance(fixed_route_points)

            for _ in range(max(1, neighborhood_iterations)):
                combined = fixed_task_points + [points[index] for index in required]
                order = nearest_neighbor_route(self.robot.current_position, combined)
                order = two_opt_open(order, self.robot.current_position, combined)
                order_position = {node: position for position, node in enumerate(order)}
                for local_index, sector_index in enumerate(required):
                    node = len(fixed_task_points) + local_index
                    position = order_position[node]
                    previous = (
                        np.asarray(self.robot.current_position, dtype=float)
                        if position == 0
                        else np.asarray(combined[order[position - 1]], dtype=float)
                    )
                    next_point = (
                        None
                        if position + 1 == len(order)
                        else np.asarray(combined[order[position + 1]], dtype=float)
                    )
                    allowed = self._cover_allowed[sector_index]
                    candidates = [points[sector_index]]
                    previous_geometry = Point(float(previous[0]), float(previous[1]))
                    candidates.append(
                        self._point_from_geometry(nearest_points(allowed, previous_geometry)[0])
                    )
                    if next_point is not None:
                        next_geometry = Point(float(next_point[0]), float(next_point[1]))
                        candidates.append(
                            self._point_from_geometry(nearest_points(allowed, next_geometry)[0])
                        )
                        segment = LineString([previous, next_point])
                        candidates.append(
                            self._point_from_geometry(nearest_points(allowed, segment)[0])
                        )

                    def local_cost(candidate):
                        value = float(np.linalg.norm(candidate - previous))
                        if next_point is not None:
                            value += float(np.linalg.norm(next_point - candidate))
                        return value

                    points[sector_index] = min(candidates, key=local_cost)

            optimized = []
            for index in required:
                point = points[index]
                if not self._cover_patches[index].difference(reception(point)).is_empty:
                    point = original_points[index]
                if not self._cover_patches[index].difference(reception(point)).is_empty:
                    raise RuntimeError("Optimized scan point lost its sector coverage certificate")
                if np.linalg.norm(point - self._cover_neighborhood_points[index]) > 1e-7:
                    self._cover_neighborhood_updates += 1
                self._cover_neighborhood_points[index] = point.copy()
                optimized.append(point)

            optimized_route_m = self._route_distance(fixed_task_points + optimized)
            self._cover_neighborhood_planned_saving_m += max(
                0.0,
                fixed_route_m - optimized_route_m,
            )
            return [Task("search", point.copy()) for point in optimized]

        def _search_measurement_count(self, chosen, remaining):
            return sum(
                remaining[channel].intersection(self._hex_disks[index]).area > 1e-8
                for index in chosen
                for channel in remaining
            )

        def _minimum_certifying_hex_subset(self, tasks, remaining=None):
            if remaining is None:
                unknown = self.unknown()
                remaining = {channel: self.map.remaining[channel] for channel in unknown}
            else:
                unknown = list(remaining)
            if not unknown:
                return []
            self._initialize_hex_cover(tasks)
            best = None
            task_points = [np.asarray(task.point, dtype=float) for task in tasks]
            for mask in range(1 << 6):
                chosen = [index for index in range(6) if mask & (1 << index)]
                if any(
                    not any(
                        remaining[channel].intersection(self._hex_disks[index]).area > 1e-8
                        for channel in unknown
                    )
                    for index in chosen
                ):
                    # A heuristic open-route score can otherwise retain a point
                    # that is geometrically redundant.  The execution layer is
                    # correct to reject such a no-progress forced scan.
                    continue
                if chosen:
                    future_cover = unary_union([self._hex_disks[index] for index in chosen])
                    certifies = all(
                        remaining[channel].difference(future_cover).is_empty
                        for channel in unknown
                    )
                else:
                    certifies = all(remaining[channel].is_empty for channel in unknown)
                if not certifies:
                    continue
                points = task_points + [self._hex_points[index] for index in chosen]
                # Action cost breaks route-length ties in favour of fewer forced
                # all-channel scans.  The actual execution still re-plans after
                # every observation.
                score = (
                    self._route_distance(points)
                    + 30.0 * self._search_measurement_count(chosen, remaining)
                )
                key = (score, len(chosen), tuple(chosen))
                if best is None or key < best[0]:
                    best = (key, chosen)
            return None if best is None else best[1]

        def _hex_plan_score(self, tasks, remaining):
            chosen = self._minimum_certifying_hex_subset(tasks, remaining)
            if chosen is None:
                return math.inf
            points = [np.asarray(task.point, dtype=float) for task in tasks]
            points.extend(self._hex_points[index] for index in chosen)
            return (
                self._route_distance(points)
                + 30.0 * self._search_measurement_count(chosen, remaining)
            )

        def _marginal_scan(self, point):
            unknown = self.unknown()
            if not unknown:
                return False
            tasks = self.target_tasks()
            self._initialize_hex_cover(tasks)
            disk = reception(point)
            remaining = {channel: self.map.remaining[channel] for channel in unknown}
            eligible = [
                channel
                for channel in unknown
                if remaining[channel].intersection(disk).area > 1e-8
            ]
            if not eligible:
                return False

            groups = {}
            for channel in eligible:
                groups.setdefault(remaining[channel].wkb, []).append(channel)
            candidate_groups = list(groups.values())
            if len(candidate_groups) > 1:
                candidate_groups.append(eligible)

            baseline_score = self._hex_plan_score(tasks, remaining)
            current_channel = self.robot.current_channel
            best = None
            for group in candidate_groups:
                hypothetical = dict(remaining)
                for channel in group:
                    hypothetical[channel] = hypothetical[channel].difference(disk)
                future_score = self._hex_plan_score(tasks, hypothetical)
                switch_count = len(group) - (1 if current_channel in group else 0)
                immediate_s = 5.0 * len(group) + switch_count
                planned_saving_s = (baseline_score - future_score) / 5.0 - immediate_s
                self._marginal_scan_evaluations += 1
                key = (planned_saving_s, -len(group))
                if best is None or key > best[0]:
                    best = (key, list(group), planned_saving_s)

            if best is None or best[2] <= marginal_scan_margin_s:
                return False
            selected = best[1]
            changed = False
            for channel in sorted(
                selected,
                key=lambda candidate: (candidate != self.robot.current_channel, candidate),
            ):
                if self.channels[channel].status != "UNKNOWN":
                    continue
                gain = self.map.gain(channel, point)
                if gain <= 1e-8:
                    continue
                self.sense(channel, point, "MARGINAL_PIGGYBACK")
                self.piggyback_measures += 1
                self.piggyback_area += gain
                changed = True
            if changed:
                self._marginal_scan_accepts += 1
                self._marginal_scan_planned_saving_s += best[2]
            return changed

        def scan(self, point, force=False):
            if force or not enable_marginal_scan:
                return super().scan(point, force=force)
            return self._marginal_scan(point)

        def planned_searches(self, tasks):
            need = self.remaining_shape()
            if need.is_empty:
                return []
            if (
                resume_dynamic_search
                and self._adaptive_resume_active()
            ):
                return super().planned_searches(tasks)
            if enable_neighborhood_cover:
                self._initialize_hex_cover(tasks)
                searches = self._optimized_neighborhood_searches(tasks)
                if searches:
                    return searches
            chosen = self._minimum_certifying_hex_subset(tasks)
            if chosen is not None:
                return [Task("search", self._hex_points[index].copy()) for index in chosen]
            return super().planned_searches(tasks)

        def run(self):
            result = super().run()
            result["hex_cover"] = {
                "ring_radius_m": ring_radius_m,
                "orientation_rad": self._hex_orientation_rad,
                "marginal_scan_evaluations": self._marginal_scan_evaluations,
                "marginal_scan_accepts": self._marginal_scan_accepts,
                "marginal_scan_planned_saving_s": self._marginal_scan_planned_saving_s,
                "neighborhood_point_updates": self._cover_neighborhood_updates,
                "neighborhood_planned_saving_m": self._cover_neighborhood_planned_saving_m,
            }
            return result

    return HexCoverStrategy


def build_strategy(
    *,
    hex_cover: bool = False,
    gain_threshold_m2: float = DEFAULT_PIGGYBACK_GAIN_M2,
    disable_piggyback: bool = False,
    disable_finish_exception: bool = False,
    ring_radius_m: float = DEFAULT_HEX_RING_RADIUS_M,
    resume_piggyback_at_known_sources: int | None = None,
    resumed_gain_threshold_m2: float = DEFAULT_PIGGYBACK_GAIN_M2,
    resume_dynamic_search: bool = False,
    resume_by_search_stop: int | None = None,
    enable_marginal_scan: bool = False,
    marginal_scan_margin_s: float = 2.0,
    enable_neighborhood_cover: bool = False,
    neighborhood_iterations: int = 4,
) -> tuple[type, tuple[Any, ...]]:
    """载入内嵌模型并按开关组装最终策略类，返回 ``(策略类, 模型元组)``。

    不带任何开关时，返回的即最终演练采用的"空频道低源覆盖"策略。
    """
    strategy_types = load_implementation()
    gain_threshold = (
        DISABLED_PIGGYBACK_GAIN_M2 if disable_piggyback else gain_threshold_m2
    )
    always_finish_channel = not (disable_finish_exception or disable_piggyback)
    if hex_cover:
        strategy_class = build_hex_cover_strategy(
            strategy_types[1],
            gain_threshold,
            always_finish_channel=always_finish_channel,
            ring_radius_m=ring_radius_m,
            resume_piggyback_at_known_sources=resume_piggyback_at_known_sources,
            resumed_gain_threshold_m2=resumed_gain_threshold_m2,
            resume_dynamic_search=resume_dynamic_search,
            resume_by_search_stop=resume_by_search_stop,
            enable_marginal_scan=enable_marginal_scan,
            marginal_scan_margin_s=marginal_scan_margin_s,
            enable_neighborhood_cover=enable_neighborhood_cover,
            neighborhood_iterations=neighborhood_iterations,
        )
    else:
        strategy_class = build_empty_channel_strategy(
            strategy_types[1],
            gain_threshold,
            always_finish_channel=always_finish_channel,
            resume_piggyback_at_known_sources=resume_piggyback_at_known_sources,
            resumed_gain_threshold_m2=resumed_gain_threshold_m2,
            resume_by_search_stop=resume_by_search_stop,
        )
    return strategy_class, strategy_types


# 说明：不再在导入时构建策略（避免副作用）；由 main.py / 本文件 main() 调
# build_strategy(...) 获取最终策略类。


# =========================================================================== #
# 本地随机测试（离线批量；robot 与 client.RobotClient 鸭子类型一致）
# =========================================================================== #
class RecordingRobot:
    """把动作原样转发给真身，同时记录时间线（只记录、不干预）。"""

    def __init__(self, inner, trace: list[dict[str, Any]]):
        self._inner = inner
        self._trace = trace

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def enter(self):
        result = self._inner.enter()
        self._trace.append({"kind": "enter", "virtual_time_s": self._inner.virtual_time_s})
        return result

    def exit(self):
        result = self._inner.exit()
        self._trace.append({"kind": "exit", "virtual_time_s": self._inner.virtual_time_s})
        return result

    def _moved(self, x: float, y: float) -> float:
        before = self._inner.current_position
        return math.hypot(float(x) - before[0], float(y) - before[1])

    def measure(self, x: float, y: float, channel: int):
        before_time = float(self._inner.virtual_time_s)
        before_channel = self._inner.current_channel
        moved = self._moved(x, y)
        result = self._inner.measure(x, y, channel)
        self._trace.append(
            {
                "kind": "measure",
                "x": float(x),
                "y": float(y),
                "channel": int(channel),
                "result": result.result,
                "svd_deg": result.svd_deg,
                "move_m": moved,
                "switched": before_channel is not None and int(channel) != before_channel,
                "duration_s": float(self._inner.virtual_time_s) - before_time,
                "virtual_time_s": float(self._inner.virtual_time_s),
            }
        )
        return result

    def clear(self, x: float, y: float, channel: int):
        before_time = float(self._inner.virtual_time_s)
        moved = self._moved(x, y)
        result = self._inner.clear(x, y, channel)
        self._trace.append(
            {
                "kind": "clear",
                "x": float(x),
                "y": float(y),
                "channel": int(channel),
                "cleared": bool(result.cleared),
                "move_m": moved,
                "duration_s": float(self._inner.virtual_time_s) - before_time,
                "virtual_time_s": float(self._inner.virtual_time_s),
            }
        )
        return result


def action_times(trace: list[dict[str, Any]]) -> dict[str, float]:
    """把动作时间线拆成移动/检测/切频/清除/尾部补盲（秒）。"""
    movement = sum(row.get("move_m", 0.0) for row in trace) / MOVE_SPEED_MPS
    measures = sum(1 for row in trace if row["kind"] == "measure")
    switches = sum(1 for row in trace if row.get("switched"))
    successes = sum(1 for row in trace if row["kind"] == "clear" and row["cleared"])
    failures = sum(1 for row in trace if row["kind"] == "clear" and not row["cleared"])
    clear_times = [
        row["virtual_time_s"] for row in trace if row["kind"] == "clear" and row["cleared"]
    ]
    total = trace[-1]["virtual_time_s"] if trace else 0.0
    return {
        "movement_time_s": movement,
        "measurement_time_s": measures * MEASURE_TIME_S,
        "switching_time_s": switches * SWITCH_TIME_S,
        "clearing_time_s": successes * CLEAR_SUCCESS_TIME_S + failures * CLEAR_FAIL_TIME_S,
        "tail_after_last_clear_s": (total - max(clear_times)) if clear_times else 0.0,
        "measure_count": float(measures),
        "clear_count": float(successes + failures),
        "clear_success_count": float(successes),
    }


def run_case(
    case_id: int,
    seed: int,
    strategy_class: type,
    *,
    num_sources: int | None = None,
    keep_trace: bool = False,
) -> dict[str, Any]:
    """在本地仿真器上跑一个案例；返回与批量汇总一致的逐案指标。"""
    simulator = LocalSimulator(seed, num_sources=num_sources)
    trace: list[dict[str, Any]] = []
    strategy = strategy_class(RecordingRobot(simulator, trace))

    error: str | None = None
    summary: dict[str, Any] = {}
    try:
        summary = dict(strategy.run() or {})
    except Exception as exc:  # noqa: BLE001 - 单案失败不终止整批
        error = repr(exc)

    source_total = int(simulator.total_sources)
    cleared = int(simulator.cleared_count)
    total_time = float(simulator.virtual_time_s)
    split = action_times(trace)
    result: dict[str, Any] = {
        "case_id": case_id,
        "seed": seed,
        "strategy": strategy_class.__name__,
        "error": error,
        "source_total": source_total,
        "source_cleared": cleared,
        "success": error is None and cleared == source_total,
        "total_virtual_time_s": total_time,
        "virtual_time_min": total_time / 60.0,
        "seconds_per_source": (total_time / source_total) if source_total else None,
        "strategy_summary": summary,
        **split,
    }
    if result["seconds_per_source"] is not None:
        result["per_source_movement_s"] = split["movement_time_s"] / source_total
        result["per_source_action_s"] = (
            split["measurement_time_s"] + split["switching_time_s"] + split["clearing_time_s"]
        ) / source_total
        result["per_source_tail_s"] = split["tail_after_last_clear_s"] / source_total
    if keep_trace:
        result["trace"] = trace
    return result


def summarize(results: list[dict[str, Any]], random_state: int) -> dict[str, Any]:
    """逐案"总时间 ÷ 真实源数"后再跨案取均值；失败案例保留参与统计。"""
    per_source = [
        float(case["seconds_per_source"])
        for case in results
        if case["seconds_per_source"] is not None
    ]
    return {
        "case_count": len(results),
        "random_state": random_state,
        "success_count": sum(1 for case in results if case["success"]),
        "success_rate": (
            sum(1 for case in results if case["success"]) / len(results) if results else 0.0
        ),
        "mean_source_count": (
            statistics.fmean(float(case["source_total"]) for case in results) if results else None
        ),
        "mean_seconds_per_source": (statistics.fmean(per_source) if per_source else None),
        "median_seconds_per_source": (statistics.median(per_source) if per_source else None),
        "p90_seconds_per_source": _percentile(per_source, 90),
        "share_within_250s": (
            sum(1 for value in per_source if value <= 250.0) / len(per_source)
            if per_source
            else None
        ),
        "mean_per_source_movement_s": _mean(results, "per_source_movement_s"),
        "mean_per_source_action_s": _mean(results, "per_source_action_s"),
        "mean_per_source_tail_s": _mean(results, "per_source_tail_s"),
        "mean_total_virtual_time_s": _mean(results, "total_virtual_time_s"),
        "errors": [case["error"] for case in results if case["error"]],
    }


def _mean(results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in results if case.get(key) is not None]
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(position // 1)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass
class _CaseRow:
    case_id: int
    seed: int
    strategy: str
    source_total: int
    source_cleared: int
    success: bool
    seconds_per_source: float | None
    error: str | None
    total_virtual_time_s: float | None = None
    virtual_time_min: float | None = None
    movement_time_s: float | None = None
    measurement_time_s: float | None = None
    switching_time_s: float | None = None
    clearing_time_s: float | None = None
    tail_after_last_clear_s: float | None = None
    measure_count: float | None = None
    clear_count: float | None = None
    clear_success_count: float | None = None
    detected: int | None = None
    empty_certified: int | None = None
    unknown: int | None = None
    per_source_movement_s: float | None = None
    per_source_action_s: float | None = None
    per_source_tail_s: float | None = None


def write_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    output_prefix: Path | str,
) -> None:
    """写 ``<prefix>.csv``（逐案，表头/取值均为英文）与 ``<prefix>.json``（汇总）。"""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in results:
        stats = case.get("strategy_summary") or {}
        rows.append(
            asdict(
                _CaseRow(
                    case_id=int(case["case_id"]),
                    seed=int(case["seed"]),
                    strategy=str(case["strategy"]),
                    source_total=int(case["source_total"]),
                    source_cleared=int(case["source_cleared"]),
                    success=bool(case["success"]),
                    seconds_per_source=case["seconds_per_source"],
                    error=case["error"],
                    total_virtual_time_s=case.get("total_virtual_time_s"),
                    virtual_time_min=case.get("virtual_time_min"),
                    movement_time_s=case.get("movement_time_s"),
                    measurement_time_s=case.get("measurement_time_s"),
                    switching_time_s=case.get("switching_time_s"),
                    clearing_time_s=case.get("clearing_time_s"),
                    tail_after_last_clear_s=case.get("tail_after_last_clear_s"),
                    measure_count=case.get("measure_count"),
                    clear_count=case.get("clear_count"),
                    clear_success_count=case.get("clear_success_count"),
                    detected=stats.get("detected"),
                    empty_certified=stats.get("empty_certified"),
                    unknown=stats.get("unknown"),
                    per_source_movement_s=case.get("per_source_movement_s"),
                    per_source_action_s=case.get("per_source_action_s"),
                    per_source_tail_s=case.get("per_source_tail_s"),
                )
            )
        )
    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_random_state(random_state: int | None) -> int:
    """未显式指定种子时，每次运行重新取一个随机种子（并写进结果便于复现）。"""
    if random_state is not None:
        return int(random_state)
    return int(np.random.default_rng().integers(0, 2**31 - 1))


def source_count_for_seed(seed: int) -> int:
    """只探测某随机种子会生成多少个干扰源，不消耗真实仿真的随机流。

    ``LocalSimulator(seed)`` 在 ``num_sources`` 未显式给出时，用同一颗种子抽一个
    ``[SOURCE_COUNT_MIN, SOURCE_COUNT_MAX]`` 的整数；这里复刻该抽样，使种子筛选
    与真实案例完全一致。
    """
    rng = np.random.default_rng(int(seed))
    return int(rng.integers(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX + 1))


def selected_seeds(
    start: int,
    count: int,
    minimum: int,
    maximum: int,
) -> list[int]:
    """从 ``start`` 起递增取种子，只保留源数落在 ``[minimum, maximum]`` 内的种子。

    与 ``--source-count`` 的区别：后者固定源数但会让源数分布偏离区间，前者保持
    题目原有的随机源数，只把区间外的种子换成区间内的种子。
    """
    if minimum > maximum:
        raise ValueError(f"种子筛选区间非法：{minimum} > {maximum}")
    seeds: list[int] = []
    seed = int(start)
    while len(seeds) < count:
        if minimum <= source_count_for_seed(seed) <= maximum:
            seeds.append(seed)
        seed += 1
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题3 策略本地随机测试（离线，不联网）",
    )
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="随机种子；不指定则每次运行都重新随机（结果里会记录本次种子）",
    )
    parser.add_argument(
        "--source-count",
        type=int,
        default=None,
        help="固定每个案例的干扰源数（例如 10）；默认在题面的 10~16 之间随机",
    )
    parser.add_argument(
        "--min-source-count",
        type=int,
        default=SOURCE_COUNT_MIN,
        help=f"筛掉源数低于此值的随机种子（默认 {SOURCE_COUNT_MIN}）",
    )
    parser.add_argument(
        "--max-source-count",
        type=int,
        default=SOURCE_COUNT_MAX,
        help=f"筛掉源数高于此值的随机种子（默认 {SOURCE_COUNT_MAX}；低源实验用 12）",
    )
    parser.add_argument("--gain-threshold", type=float, default=DEFAULT_PIGGYBACK_GAIN_M2)
    parser.add_argument(
        "--disable-finish-exception",
        action="store_true",
        help="关闭『这次测点即可测完该频道剩余区域则必测』的规则。",
    )
    parser.add_argument(
        "--hex-cover",
        action="store_true",
        help="启用认证的圆心＋六边形覆盖骨架。",
    )
    parser.add_argument(
        "--disable-piggyback",
        action="store_true",
        help="仅在强制搜索点批量覆盖未知频道。",
    )
    parser.add_argument("--ring-radius", type=float, default=DEFAULT_HEX_RING_RADIUS_M)
    parser.add_argument(
        "--resume-piggyback-at-known-sources",
        type=int,
        default=None,
        help="确认该数量源后恢复选择性顺带覆盖。",
    )
    parser.add_argument("--resumed-gain-threshold", type=float, default=DEFAULT_PIGGYBACK_GAIN_M2)
    parser.add_argument(
        "--resume-dynamic-search",
        action="store_true",
        help="恢复自适应模式时同时恢复父类残差搜索规划。",
    )
    parser.add_argument(
        "--resume-by-search-stop",
        type=int,
        default=None,
        help="仅当已知源触发发生在这个搜索次数之前才锁定自适应模式。",
    )
    parser.add_argument(
        "--marginal-scan",
        action="store_true",
        help="仅当重算后的认证规划更便宜时才接受顺带覆盖。",
    )
    parser.add_argument("--marginal-scan-margin", type=float, default=2.0)
    parser.add_argument(
        "--neighborhood-cover",
        action="store_true",
        help="在外扇区认证可行域内选择每个覆盖测点。",
    )
    parser.add_argument("--neighborhood-iterations", type=int, default=4)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_HERE / "outputs/tables/q3_strategy_local",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source_count is not None and not (
        SOURCE_COUNT_MIN <= args.source_count <= CHANNEL_MAX
    ):
        raise SystemExit(
            f"--source-count 必须在 {SOURCE_COUNT_MIN}..{CHANNEL_MAX} 之间"
        )
    strategy_class, _ = build_strategy(
        hex_cover=args.hex_cover,
        gain_threshold_m2=args.gain_threshold,
        disable_piggyback=args.disable_piggyback,
        disable_finish_exception=args.disable_finish_exception,
        ring_radius_m=args.ring_radius,
        resume_piggyback_at_known_sources=args.resume_piggyback_at_known_sources,
        resumed_gain_threshold_m2=args.resumed_gain_threshold,
        resume_dynamic_search=args.resume_dynamic_search,
        resume_by_search_stop=args.resume_by_search_stop,
        enable_marginal_scan=args.marginal_scan,
        marginal_scan_margin_s=args.marginal_scan_margin,
        enable_neighborhood_cover=args.neighborhood_cover,
        neighborhood_iterations=args.neighborhood_iterations,
    )
    random_state = resolve_random_state(args.random_state)
    source_hint = (
        f"每个案例固定 {args.source_count} 个源"
        if args.source_count is not None
        else "源数在 10~16 随机"
    )
    print(
        f"随机状态 {random_state}（{args.cases} 个案例，{source_hint}；"
        f"复现本次运行请加 --random-state {random_state}）"
    )
    seeds = [random_state + index for index in range(args.cases)]
    results = [
        run_case(index + 1, seed, strategy_class, num_sources=args.source_count)
        for index, seed in enumerate(seeds)
    ]
    # 逐案例明细（与 problem4 本地批量同版式）
    for result in results:
        stats = result.get("strategy_summary") or {}
        per_source = result["seconds_per_source"]
        per_source_text = (
            f"{float(per_source):8.2f} s" if per_source is not None else "     n/a"
        )
        action_min = (
            float(result["measurement_time_s"])
            + float(result["switching_time_s"])
            + float(result["clearing_time_s"])
        ) / 60.0
        print(
            f"[{int(result['case_id']):>2}] seed {int(result['seed'])}"
            f"｜源 {int(result['source_total'])}"
            f"（检测 {stats.get('detected', '-')} / "
            f"空证书 {stats.get('empty_certified', '-')} / "
            f"未决 {stats.get('unknown', '-')}）"
            f"｜清除 {int(result['source_cleared'])}/{int(result['source_total'])}"
            f"｜单源平均 {per_source_text}"
            f"｜总 {float(result['total_virtual_time_s']) / 60.0:7.2f} min"
            f"｜移动 {float(result['movement_time_s']) / 60.0:7.2f} min"
            f"｜动作 {action_min:7.2f} min"
        )
    summary = summarize(results, random_state)
    summary["strategy"] = (
        "q3_empty_channel_hex_cover" if args.hex_cover
        else "q3_empty_channel_piggyback_threshold"
    )
    summary["gain_threshold_m2"] = args.gain_threshold
    summary["hex_cover"] = args.hex_cover
    summary["ring_radius_m"] = args.ring_radius
    summary["seeds"] = seeds
    per_source_values = [
        float(result["seconds_per_source"])
        for result in results
        if result["seconds_per_source"] is not None
    ]
    mean_s = float(summary["mean_seconds_per_source"] or 0.0)
    median_s = float(summary["median_seconds_per_source"] or 0.0)
    worst_s = max(per_source_values) if per_source_values else 0.0
    print(
        f"合计：{len(results)} 例，成功 {summary['success_count']}/{len(results)}；"
        f"单源平均 {mean_s:.2f} s"
        f"（中位 {median_s:.2f} s，最差 {worst_s:.2f} s）"
    )
    write_results(results, summary, args.output_prefix)
    print(f"逐案结果(CSV)：{args.output_prefix.with_suffix('.csv')}")
    print(f"汇总(JSON)：{args.output_prefix.with_suffix('.json')}")
    # 完整汇总已写入 <output-prefix>.json；如仍需打印，可取消下一行注释：
    # print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
