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
