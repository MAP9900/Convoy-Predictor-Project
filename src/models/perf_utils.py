#Imports
import time, functools, psutil, os, tracemalloc

#Performance Decorator Function
def track_performance(label=None, peak_py=False):
    def decorator(func):
        name = label or func.__name__
        proc = psutil.Process()
        ncpu = os.cpu_count() or 1
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if peak_py:
                tracemalloc.start()
            t0 = time.perf_counter()
            u0, s0 = proc.cpu_times().user, proc.cpu_times().system
            ru0 = sum(c.cpu_times().user for c in proc.children(recursive=True))
            rs0 = sum(c.cpu_times().system for c in proc.children(recursive=True))
            m0 = proc.memory_info().rss
            out = func(*args, **kwargs)
            t1 = time.perf_counter()
            u1, s1 = proc.cpu_times().user, proc.cpu_times().system
            ru1 = sum(c.cpu_times().user for c in proc.children(recursive=True))
            rs1 = sum(c.cpu_times().system for c in proc.children(recursive=True))
            m1 = proc.memory_info().rss
            dur = t1 - t0
            cpu_time = (u1 - u0) + (s1 - s0) + (ru1 - ru0) + (rs1 - rs0)
            cpu_pct = 100.0 * cpu_time / (dur * ncpu) if dur > 0 else 0.0
            dmb = (m1 - m0) / (1024 ** 2)
            if peak_py:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                pmb = peak / (1024 ** 2)
                print(f"\nPerformance Stats:\n{name} completed in {dur:.2f}s | ΔRSS {dmb:.2f} MB | CPU {cpu_pct:.1f}% | PyPeak {pmb:.2f} MB\n")
            else:
                print(f"\nPerformance Stats:\n{name} completed in {dur:.2f}s | ΔRSS {dmb:.2f} MB | CPU {cpu_pct:.1f}%\n")
            return out
        return wrapper
    return decorator


# --- Example Usage ---

# from src.models.perf_utils import track_performance

# @track_performance("cnb.optimize")
# def run_cnb_opt():
#     return cnb2.optimize(scoring=recall_scorers, refit="recall_pos")

# run_cnb_opt()