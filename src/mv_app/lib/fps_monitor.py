import time
import numpy as np
from .utils import (
    ValidationError, MsgType,
)

class MovingWindowAverage:

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.alpha = 2 / (window_size + 1)
        self.mean = -1

    def update(self, new_value):
        # Initialize with the very first value received
        if self.mean == -1:
            self.mean = float(new_value)
        else:
            # Apply the Exponential Moving Average formula
            self.mean = (self.alpha * new_value) + ((1 - self.alpha) * self.mean)
        
        return self.mean

class MovingMean:

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.window = np.array([np.nan for _ in range(window_size)], dtype=np.float32)
        self.idx = 0

    @property
    def mean(self):

        if self.idx < self.window_size:
            return np.mean(self.window[~np.isnan(self.window)])

        return np.mean(self.window)

    def update(self, value):

        if self.idx < self.window_size:
            self.window[self.idx] = value
            self.idx += 1
        else:
            self.window[:-1] = self.window[1:]
            self.window[-1] = value

        return self.mean

class FPSMonitor:
    
    def __init__(self):
        
        self.target_fps:float = None
        self._target_period:float = None
        self._last_ts:float = time.perf_counter()

        self._moving_mean_fps = MovingMean()
        self._fps:float = 0

    def set_target_fps(self, target_fps:float):
        if target_fps <= 0:
            raise ValidationError("Target FPS must be greater than 0", MsgType.ERROR)

        self.target_fps = float(target_fps)
        self._target_period = 1.0 / self.target_fps
        self._last_ts = time.perf_counter()

    def tick(self):

        if self.target_fps is not None:
            remaining_time = self._target_period - (time.perf_counter() - self._last_ts)
            if remaining_time > 0:
                time.sleep(remaining_time)
        
        new_ts = time.perf_counter()
        fps = 1.0 / (new_ts - self._last_ts)
        self._last_ts = new_ts
        self._fps = self._moving_mean_fps.update(fps)
        
    @property
    def fps(self):
        return self._fps
