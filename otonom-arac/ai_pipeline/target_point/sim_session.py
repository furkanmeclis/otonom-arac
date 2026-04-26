"""Deterministic simulator session helpers for scripted target-point runs."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SimObservation:
    image: np.ndarray
    reward: float
    done: bool
    pos: Tuple[float, float, float]
    cte: float
    speed: float
    forward_vel: float
    hit: str
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    lap_count: int
    last_lap_time: float
    info: Dict[str, Any]


class DonkeySimSession:
    """Thin deterministic wrapper around the Donkey simulator gym env."""

    def __init__(
        self,
        cfg,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
        start_delay: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.env_name = env_name or getattr(cfg, "DONKEY_GYM_ENV_NAME")
        self.seed = seed
        self.start_delay = start_delay
        self.env = None
        self.step_index = 0
        self.elapsed_seconds = 0.0
        self.dt = 1.0 / max(1.0, float(getattr(cfg, "DRIVE_LOOP_HZ", 20.0)))

    def _build_conf(self) -> Dict[str, Any]:
        conf = copy.deepcopy(dict(getattr(self.cfg, "GYM_CONF", {})))
        conf["exe_path"] = getattr(self.cfg, "DONKEY_SIM_PATH", "remote")
        conf["host"] = getattr(self.cfg, "SIM_HOST", "127.0.0.1")
        conf["port"] = int(conf.get("port", getattr(self.cfg, "DONKEY_SIM_PORT", 9091)))
        conf["guid"] = int(conf.get("guid", 0))
        conf["frame_skip"] = int(conf.get("frame_skip", 1))
        conf["start_delay"] = float(
            conf.get(
                "start_delay",
                self.start_delay if self.start_delay is not None else getattr(self.cfg, "DONKEY_SIM_START_DELAY", 5.0),
            )
        )
        conf["cam_resolution"] = tuple(
            conf.get(
                "cam_resolution",
                (
                    int(getattr(self.cfg, "IMAGE_H", 120)),
                    int(getattr(self.cfg, "IMAGE_W", 160)),
                    int(getattr(self.cfg, "IMAGE_DEPTH", 3)),
                ),
            )
        )
        conf.setdefault(
            "cam_config",
            {
                "img_w": int(getattr(self.cfg, "IMAGE_W", 160)),
                "img_h": int(getattr(self.cfg, "IMAGE_H", 120)),
                "img_d": int(getattr(self.cfg, "IMAGE_DEPTH", 3)),
                "img_enc": "JPG",
                "fov": 60,
            },
        )
        return conf

    def open(self) -> "DonkeySimSession":
        if self.env is not None:
            return self

        if self.seed is not None:
            random.seed(int(self.seed))
            np.random.seed(int(self.seed))

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import gym  # imported lazily to avoid side effects in training code paths
        import gym_donkeycar  # noqa: F401

        self.env = gym.make(self.env_name, conf=self._build_conf())
        if self.seed is not None and hasattr(self.env, "seed"):
            self.env.seed(int(self.seed))

        self.step_index = 0
        self.elapsed_seconds = 0.0
        return self

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self) -> "DonkeySimSession":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> np.ndarray:
        if self.env is None:
            self.open()

        self.step_index = 0
        self.elapsed_seconds = 0.0
        return np.asarray(self.env.reset())

    def step(self, steering: float, throttle: float, brake: float = 0.0) -> SimObservation:
        if self.env is None:
            self.open()

        steering = float(steering)
        throttle = 0.0 if float(brake) > 0.0 else float(throttle)
        image, reward, done, info = self.env.step(np.asarray([steering, throttle], dtype=np.float32))

        roll_deg, pitch_deg, yaw_deg = (0.0, 0.0, 0.0)
        car = info.get("car")
        if isinstance(car, (tuple, list)) and len(car) == 3:
            roll_deg, pitch_deg, yaw_deg = (float(car[0]), float(car[1]), float(car[2]))

        self.step_index += 1
        self.elapsed_seconds = self.step_index * self.dt

        return SimObservation(
            image=np.asarray(image),
            reward=float(reward),
            done=bool(done),
            pos=tuple(float(value) for value in info.get("pos", (0.0, 0.0, 0.0))),
            cte=float(info.get("cte", 0.0)),
            speed=float(info.get("speed", 0.0)),
            forward_vel=float(info.get("forward_vel", 0.0)),
            hit=str(info.get("hit", "none")),
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            lap_count=int(info.get("lap_count", 0)),
            last_lap_time=float(info.get("last_lap_time", 0.0)),
            info=dict(info),
        )

    def prime(self) -> SimObservation:
        """Reset the env and fetch the first telemetry-bearing observation."""
        self.reset()
        return self.step(0.0, 0.0, 0.0)
