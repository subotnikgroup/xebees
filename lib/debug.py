import xp
from time import perf_counter as time
from functools import wraps
from contextlib import contextmanager

def prms(A,B, label=""):
    print(label, xp.sqrt(xp.mean((A-B)**2)))

@contextmanager
def timer_ctx(label=""):
   start = time()
   yield
   end = time()
   elapsed = end - start
   print(f"[{label}] Elapsed time: {1e6*elapsed:0.3}us")

def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        elapsed = end - start
        print(f"Elapsed time: {elapsed}s")
        return result
    return wrapper


from typing import Protocol


class Timer(Protocol):
    def start(self):
        pass

    def stop(self):
        pass
        """
        Blocks execution until everything before it has completed. Returns the
        duration since the last call to start(), in milliseconds.
        """


class CuPyNumericTimer(Timer):
    def __init__(self):
        self._start_time = None

    def start(self):
        from legate.timing import time

        self._start_time = time("us")

    def stop(self):
        from legate.timing import time

        end_future = time("us")
        return (end_future - self._start_time) / 1000.0


class CuPyTimer(Timer):
    def __init__(self):
        self._start_event = None

    def start(self):
        from cupy import cuda

        self._start_event = cuda.Event()
        self._start_event.record()

    def stop(self):
        from cupy import cuda

        end_event = cuda.Event()
        end_event.record()
        end_event.synchronize()
        return cuda.get_elapsed_time(self._start_event, end_event)


class NumPyTimer(Timer):
    def __init__(self):
        self._start_time = None

    def start(self):
        from time import perf_counter_ns

        self._start_time = perf_counter_ns() / 1000.0

    def stop(self):
        from time import perf_counter_ns

        end_time = perf_counter_ns() / 1000.0
        return (end_time - self._start_time) / 1000.0
