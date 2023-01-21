from time import perf_counter_ns

import numpy as np


def measure_time(fn, min_samples=10, max_samples=100, max_time_in_sec=100.0):
    """Measure time to run fn. Returns the elapsed time each run."""

    t = []
    for i in range(max_samples):
        if sum(t) / 1e9 >= max_time_in_sec and i >= min_samples:
            break
        t.append(-perf_counter_ns())
        try:
            ans = fn()
        except Exception as e:
            print(e)
            return np.array([np.nan])
        t[-1] += perf_counter_ns()
        del ans
    print(".", end="")
    return np.array(t) / 1e9
