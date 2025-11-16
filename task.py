import random
import timeit

from core.algorithms import insertion_sort, merge_sort, tim_sort

cases = [10, 100, 1_000, 10_000, 25_000, 50_000, 100_000]

if __name__ == "__main__":
    for case in cases:
        data = [random.uniform(0, case) for _ in range(case)]
        print(f"Collection size: {case}")
        print(f"{'Insertion Sort':<20}{timeit.timeit(lambda: insertion_sort(data.copy()), number=100):.6f}")
        print(f"{'Merge Sort':<20}{timeit.timeit(lambda: merge_sort(data.copy()), number=100):.6f}")
        print(f"{'Tim Sort':<20}{timeit.timeit(lambda: tim_sort(data.copy()), number=100):.6f}")
        print(f"{'Internal Sort':<20}{timeit.timeit(lambda: (data.copy().sort()), number=100):.6f}")
        print(f"{'Internal Sorted':<20}{timeit.timeit(lambda: (sorted(data.copy())), number=100):.6f}")
        print("-" * 40)
