"""机器狗与无线电干扰源环境模拟器的通信客户端。

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
