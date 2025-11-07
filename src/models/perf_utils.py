"""
perf_utils.py

Simple decorator helpers for timing model routines while keeping the codebase lightweight.
"""

#Imports
import functools
import time
import sys
from typing import Optional
try:
    import psutil
except ImportError:
    psutil = None


def _rss_bytes() -> Optional[float]:
    """
    Return current resident memory usage in bytes when possible.
    Falls back to resource.getrusage on POSIX if psutil is unavailable.
    """
    if psutil is not None:
        return psutil.Process().memory_info().rss
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage
    except Exception:
        return None


def track_perf(label: Optional[str] = None):
    """
    Decorator for logging wall-clock time and memory delta of a function call.

    Example:
        @track_perf("rf_grid_search")
        def run_grid():
            ...
    """

    def decorator(func):
        pretty_label = label or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            start_mem = _rss_bytes()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            end_mem = _rss_bytes()

            duration = end_time - start_time
            msg = f"{pretty_label} completed in {duration:.2f}s"

            if start_mem is not None and end_mem is not None:
                delta_mb = (end_mem - start_mem) / (1024 ** 2)
                msg += f" | ΔRSS {delta_mb:.2f} MB"

            if psutil is not None:
                cpu = psutil.Process().cpu_percent(interval=None)
                msg += f" | CPU {cpu:.1f}%"
            print(msg)
            return result
        return wrapper
    return decorator


__all__ = ["track_perf"]



# from src.models.perf_utils import track_perf

# @track_perf("cnb.optimize")
# def run_cnb_opt():
#     return cnb2.optimize(scoring=recall_scorers, refit="recall_pos")

# run_cnb_opt()