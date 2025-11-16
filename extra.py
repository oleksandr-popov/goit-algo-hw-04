"""Simple benchmark runner for sorting algorithms in the algorithms module.

This script uses only Python standard library modules and keeps the logic
simple while being more flexible than the original one-off prints.

Features:
- Auto-discovers sort functions inside `algorithms` (functions with 'sort' in
  their name).
- Adapts to different function signatures (e.g. quick_sort(collection, low, high)).
- Uses `timeit.repeat` and `statistics` to show min and mean timings.
- Uses `argparse` to allow running a subset of algorithms or custom sizes.
"""

import argparse
import importlib
import inspect
import random
import statistics
import timeit
from typing import Callable, List


from core import algorithms


def generate_random_integers(
    n: int, min_value: int = 0, max_value: int = 100
) -> List[int]:
    """Return list of n random integers between min_value and max_value (inclusive)."""
    return [random.randint(min_value, max_value) for _ in range(n)]


def choose_iterations(n: int, override: int | None = None) -> int:
    """Pick a reasonable number of iterations for timing based on input size.

    For small inputs we repeat many times; for large inputs we use fewer
    iterations so the script stays responsive.
    """
    if override is not None:
        return max(1, int(override))
    if n <= 50:
        return 2000
    if n <= 500:
        return 500
    if n <= 5000:
        return 50
    return 10


def make_caller(fn: Callable) -> Callable[[List[int]], List[int]]:
    """Return a zero-argument callable that runs the sort on a copy of data.

    The returned callable accepts a single parameter (collection) when used
    inside this script; here we adapt to fn's expected signature so we can call
    it uniformly.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters)

    def caller(coll: List[int]):
        # If function expects only one argument (collection), call directly.
        if len(params) == 1:
            return fn(coll)
        # If function expects (collection, low, high) like quick_sort
        if len(params) >= 3:
            try:
                return fn(coll, 0, len(coll) - 1)
            except TypeError:
                # Fall back to calling with just the collection
                return fn(coll)
        # For anything else, try calling with just the collection
        return fn(coll)

    return caller


def benchmark(
    fn: Callable[[List[int]], List[int]],
    data: List[int],
    iterations: int = 100,
    repeat: int = 3,
) -> tuple[float, float]:
    """Benchmark a sort function using timeit.repeat.

    Returns (min_time_per_call, mean_time_per_call).
    """
    # timeit.repeat returns total time for `number=iterations` runs per entry.
    times = timeit.repeat(lambda: fn(data.copy()), repeat=repeat, number=iterations)
    # Convert to per-call times
    per_call = [t / iterations for t in times]
    return min(per_call), statistics.mean(per_call)


def discover_sort_functions():
    """Discover candidate sorting functions in algorithms module.

    We select callables whose name contains 'sort' but skip helper functions
    and clearly unsafe ones like 'sleep_sort'.
    """
    funcs = {}
    for name in dir(algorithms):
        if name.startswith("_"):
            continue
        # Only include top-level sort functions (names that end with 'sort').
        if not name.endswith("sort"):
            continue
        if name == "sleep_sort":
            # skip functions that rely on sleeping threads
            continue
        obj = getattr(algorithms, name)
        if callable(obj):
            funcs[name] = obj
    return funcs


def run():
    parser = argparse.ArgumentParser(
        description="Benchmark sorting algorithms from the algorithms module"
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[10, 100, 1000, 10000],
        help="List of input sizes to test",
    )
    parser.add_argument(
        "--only", nargs="*", help="Names of algorithms to run (default: all discovered)"
    )
    parser.add_argument("--iterations", type=int, help="Override iterations per timing")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="How many repeat measurements to take (timeit.repeat)",
    )
    args = parser.parse_args()

    funcs = discover_sort_functions()
    if args.only:
        wanted = {name: funcs[name] for name in args.only if name in funcs}
    else:
        wanted = funcs

    if not wanted:
        print("No sorting functions found to benchmark.")
        return

    print(f"Found {len(wanted)} sorting functions: {', '.join(sorted(wanted))}\n")

    for n in args.sizes:
        data = generate_random_integers(n)
        print("-" * 40)
        print(f"Total elements: {n}")
        iterations = choose_iterations(n, args.iterations)
        for name, fn in sorted(wanted.items()):
            caller = make_caller(fn)
            try:
                min_t, mean_t = benchmark(
                    caller, data, iterations=iterations, repeat=args.repeat
                )
                print(
                    f"{name:20} min: {min_t:.6f}s  mean: {mean_t:.6f}s (per call, {iterations} iters)"
                )
            except Exception as exc:
                print(f"{name:20} ERROR: {exc}")
        print("*" * 40)


if __name__ == "__main__":
    run()
