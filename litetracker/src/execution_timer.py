from threading import Lock, Thread
import time
import torch
import numpy as np
import json
from pprint import pprint


class SingletonMeta(type):
    """
    This is a thread-safe implementation of the ExecutionTimer Singleton.
    Code template is taken from: https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ExecutionTimer(metaclass=SingletonMeta):
    execution_times: dict = {}
    initial_index: int = 10

    @staticmethod
    def clear():
        ExecutionTimer.execution_times = {}

    @staticmethod
    def median() -> dict:
        return {
            k: np.median(v[ExecutionTimer.initial_index :])
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def mean() -> dict:
        return {
            k: np.mean(v[ExecutionTimer.initial_index :])
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def std() -> dict:
        return {
            k: np.std(v[ExecutionTimer.initial_index :])
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def min() -> dict:
        return {
            k: np.min(v[ExecutionTimer.initial_index :])
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def max() -> dict:
        return {
            k: np.max(v[ExecutionTimer.initial_index :])
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def percentile(p: int) -> dict:
        return {
            k: np.percentile(v[ExecutionTimer.initial_index :], p)
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def len() -> dict:
        return {
            k: len(v[ExecutionTimer.initial_index :])
            for k, v in ExecutionTimer.execution_times.items()
        }

    @staticmethod
    def stats() -> dict:
        return {
            "len": ExecutionTimer.len(),
            "min": ExecutionTimer.min(),
            "max": ExecutionTimer.max(),
            "mean": ExecutionTimer.mean(),
            "std": ExecutionTimer.std(),
            "median": ExecutionTimer.median(),
            "P99": ExecutionTimer.percentile(99),
            "P95": ExecutionTimer.percentile(95),
        }

    @staticmethod
    def print_stats(mantis: int = 2):
        stats = ExecutionTimer.stats()
        # Change the precision of the floating point numbers
        for key in stats:
            stats[key] = {k: round(v, mantis) for k, v in stats[key].items()}

        headline = "#" * 8 + " EXECUTION TIME STATS (ms) " + "#" * 8
        print("\n", headline)
        print(json.dumps(stats, indent=4))
        print("#" * len(headline), "\n")

    @staticmethod
    def dump_stats_to_json(filename: str = "execution_timer.json"):
        stats = ExecutionTimer.stats()
        with open(filename, "w") as f:
            json.dump(stats, f, indent=4)


class LogExecutionTime:
    r"""
    This is a context manager class that logs the execution time of a code block.
    Example usage:

        ```python
        with LogExecutionTime("sample_block"):
            time.sleep(1)  # Simulating a time-consuming task
        ```
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        # Synchronize before starting the timer if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        # Synchronize again after the code block finishes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        delta_time = end_time - self.start_time  # Execution time in seconds
        delta_time = delta_time * 1000  # Convert to milliseconds

        # Log the execution time in the global state as a new element in a vector
        if self.name not in ExecutionTimer.execution_times:
            ExecutionTimer.execution_times[self.name] = [delta_time]
        else:
            ExecutionTimer.execution_times[self.name].append(delta_time)


if __name__ == "__main__":

    with LogExecutionTime("sample_chunk 1"):
        a = torch.randn(8, 128, 128).cuda()
        s = a.softmax(dim=1)
        ss = a.softmax(dim=2)

    with LogExecutionTime("sample_chunk 2"):
        a = torch.randn(8, 128, 128).cuda()
        s = a.softmax(dim=1)
        ss = a.softmax(dim=2)

    ExecutionTimer.print_stats()
    # ExecutionTimer.dump_stats_to_json()
