"""Sorting algorithm implementations.

This module provides a collection of in-place and out-of-place sorting
algorithms implemented for educational and testing purposes. Algorithms
include quick sort, merge sort, heap sort, radix/counting/bucket sorts,
insertion/selection/bubble sorts and several less common algorithms.

Note: Some algorithms expect non-negative integers (e.g. counting_sort,
radix_sort, pigeonhole_sort, sleep_sort) and some functions operate on
subranges of the provided list (e.g. quick_sort, bitonic_sort).
"""


def partition(collection, low, high):
    """Partitions the collection around a pivot for quicksort.

    Args:
        collection (list): The list to be partitioned.
        low (int): The starting index of the segment to be partitioned.
        high (int): The ending index of the segment to be partitioned.
    Returns:
        int: The index of the pivot after partitioning.
    """
    pivot = collection[high]
    i = low - 1
    for j in range(low, high):
        if collection[j] <= pivot:
            i += 1
            collection[i], collection[j] = collection[j], collection[i]
    collection[i + 1], collection[high] = collection[high], collection[i + 1]
    return i + 1


def quick_sort(collection, low, high):
    """Sorts a collection in place using the quicksort algorithm.

    Args:
        collection (list): The list to be sorted.
        low (int): The starting index of the segment to be sorted.
        high (int): The ending index of the segment to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(collection) <= 1:
        return collection
    if low < high:
        pivot_index = partition(collection, low, high)
        quick_sort(collection, low, pivot_index - 1)
        quick_sort(collection, pivot_index + 1, high)
    return collection


def selection_sort(collection):
    """Sorts a collection in place using the selection sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if collection[j] < collection[min_index]:
                min_index = j
        collection[i], collection[min_index] = collection[min_index], collection[i]
    return collection


def bubble_sort(collection):
    """Sorts a collection in place using the bubble sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    for i in range(n):
        for j in range(0, n - i - 1):
            if collection[j] > collection[j + 1]:
                collection[j], collection[j + 1] = collection[j + 1], collection[j]
    return collection


def merge_sort(collection):
    """Sorts a collection using the merge sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(collection) <= 1:
        return collection

    mid = len(collection) // 2
    left_half = merge_sort(collection[:mid])
    right_half = merge_sort(collection[mid:])

    return merge(left_half, right_half)


def merge(left, right):
    """Merges two sorted lists into one sorted list.

    Args:
        left (list): The first sorted list.
        right (list): The second sorted list.
    Returns:
        list: The merged sorted list.
    """
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


def radix_sort(collection):
    """Sorts a collection using the radix sort algorithm.

    Args:
        collection (list): The list of non-negative integers to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(collection) == 0:
        return collection

    max_num = max(collection)
    exp = 1
    while max_num // exp > 0:
        counting_by_digit(collection, exp)
        exp *= 10
    return collection


def counting_by_digit(collection, exp):
    """A helper function for radix sort that performs counting sort based on a specific digit.

    Args:
        collection (list): The list to be sorted.
        exp (int): The exponent corresponding to the digit to sort by.
    """
    n = len(collection)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = (collection[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (collection[i] // exp) % 10
        output[count[index] - 1] = collection[i]
        count[index] -= 1

    for i in range(n):
        collection[i] = output[i]


def insertion_sort(collection):
    """Sorts a collection in place using the insertion sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    for i in range(1, len(collection)):
        key = collection[i]
        j = i - 1
        while j >= 0 and key < collection[j]:
            collection[j + 1] = collection[j]
            j -= 1
        collection[j + 1] = key
    return collection


def heap_sort(collection):
    """Sorts a collection in place using the heap sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)

    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    for i in range(n // 2 - 1, -1, -1):
        heapify(collection, n, i)

    for i in range(n - 1, 0, -1):
        collection[i], collection[0] = collection[0], collection[i]
        heapify(collection, i, 0)

    return collection


def shell_sort(collection):
    """Sorts a collection in place using the shell sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = collection[i]
            j = i
            while j >= gap and collection[j - gap] > temp:
                collection[j] = collection[j - gap]
                j -= gap
            collection[j] = temp
        gap //= 2

    return collection


def counting_sort(collection):
    """Sorts a collection using the counting sort algorithm.

    Args:
        collection (list): The list of non-negative integers to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(collection) == 0:
        return collection

    max_val = max(collection)
    count = [0] * (max_val + 1)

    for num in collection:
        count[num] += 1

    sorted_index = 0
    for i in range(len(count)):
        while count[i] > 0:
            collection[sorted_index] = i
            sorted_index += 1
            count[i] -= 1

    return collection


def bucket_sort(collection, bucket_size=5):
    """Sorts a collection using the bucket sort algorithm.

    Args:
        collection (list): The list to be sorted.
        bucket_size (int): The size of each bucket.
    Returns:
        list: The sorted list.
    """
    if len(collection) == 0:
        return collection

    min_value = min(collection)
    max_value = max(collection)

    bucket_count = (max_value - min_value) // bucket_size + 1
    buckets = [[] for _ in range(bucket_count)]

    for num in collection:
        index = (num - min_value) // bucket_size
        buckets[index].append(num)

    sorted_collection = []
    for bucket in buckets:
        insertion_sort(bucket)
        sorted_collection.extend(bucket)

    return sorted_collection


def comb_sort(collection):
    """Sorts a collection in place using the comb sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    gap = n
    shrink = 1.3
    sorted = False

    while not sorted:
        gap = int(gap / shrink)
        if gap < 1:
            gap = 1
        sorted = True

        for i in range(n - gap):
            if collection[i] > collection[i + gap]:
                collection[i], collection[i + gap] = collection[i + gap], collection[i]
                sorted = False

    return collection


def gnome_sort(collection):
    """Sorts a collection in place using the gnome sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    index = 0
    n = len(collection)

    while index < n:
        if index == 0 or collection[index] >= collection[index - 1]:
            index += 1
        else:
            collection[index], collection[index - 1] = (
                collection[index - 1],
                collection[index],
            )
            index -= 1

    return collection


def cocktail_sort(collection):
    """Sorts a collection in place using the cocktail sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False

        for i in range(start, end):
            if collection[i] > collection[i + 1]:
                collection[i], collection[i + 1] = collection[i + 1], collection[i]
                swapped = True

        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if collection[i] > collection[i + 1]:
                collection[i], collection[i + 1] = collection[i + 1], collection[i]
                swapped = True

        start += 1

    return collection


def odd_even_sort(collection):
    """Sorts a collection in place using the odd-even sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    sorted = False

    while not sorted:
        sorted = True

        for i in range(1, n - 1, 2):
            if collection[i] > collection[i + 1]:
                collection[i], collection[i + 1] = collection[i + 1], collection[i]
                sorted = False

        for i in range(0, n - 1, 2):
            if collection[i] > collection[i + 1]:
                collection[i], collection[i + 1] = collection[i + 1], collection[i]
                sorted = False

    return collection


def pancake_sort(collection):
    """Sorts a collection in place using the pancake sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """

    def flip(sub_collection, k):
        start = 0
        while start < k:
            sub_collection[start], sub_collection[k] = (
                sub_collection[k],
                sub_collection[start],
            )
            start += 1
            k -= 1

    n = len(collection)
    for curr_size in range(n, 1, -1):
        max_index = collection.index(max(collection[:curr_size]))
        if max_index != curr_size - 1:
            if max_index != 0:
                flip(collection, max_index)
            flip(collection, curr_size - 1)

    return collection


def bitonic_sort(collection, low=0, cnt=None, ascending=True):
    """Sorts a collection using the bitonic sort algorithm.

    Args:
        collection (list): The list to be sorted.
        low (int): The starting index of the segment to be sorted.
        cnt (int): The number of elements to be sorted.
        ascending (bool): True for ascending order, False for descending order.
    Returns:
        list: The sorted list.
    """
    if cnt is None:
        cnt = len(collection)

    if cnt > 1:
        k = cnt // 2
        bitonic_sort(collection, low, k, True)
        bitonic_sort(collection, low + k, k, False)
        bitonic_merge(collection, low, cnt, ascending)

    return collection


def bitonic_merge(collection, low, cnt, ascending):
    """A helper function for bitonic sort that merges two bitonic sequences.

    Args:
        collection (list): The list to be merged.
        low (int): The starting index of the segment to be merged.
        cnt (int): The number of elements to be merged.
        ascending (bool): True for ascending order, False for descending order.
    """
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            if (ascending and collection[i] > collection[i + k]) or (
                not ascending and collection[i] < collection[i + k]
            ):
                collection[i], collection[i + k] = collection[i + k], collection[i]
        bitonic_merge(collection, low, k, ascending)
        bitonic_merge(collection, low + k, k, ascending)


def tim_sort(collection):
    """Sorts a collection using the Timsort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    min_run = 32
    n = len(collection)

    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        insertion_range(collection, start, end)

    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))

            if mid < right:
                merge_ranges(collection, left, mid, right)
        size *= 2

    return collection


def insertion_range(collection, left, right):
    """A helper function for Timsort that performs insertion sort on a specific range.

    Args:
        collection (list): The list to be sorted.
        left (int): The starting index of the range.
        right (int): The ending index of the range.
    """
    for i in range(left + 1, right + 1):
        key = collection[i]
        j = i - 1
        while j >= left and collection[j] > key:
            collection[j + 1] = collection[j]
            j -= 1
        collection[j + 1] = key


def merge_ranges(collection, left, mid, right):
    """A helper function for Timsort that merges two sorted ranges.

    Args:
        collection (list): The list to be merged.
        left (int): The starting index of the first range.
        mid (int): The ending index of the first range.
        right (int): The ending index of the second range.
    """
    left_copy = collection[left : mid + 1]
    right_copy = collection[mid + 1 : right + 1]

    left_index, right_index = 0, 0
    sorted_index = left

    while left_index < len(left_copy) and right_index < len(right_copy):
        if left_copy[left_index] <= right_copy[right_index]:
            collection[sorted_index] = left_copy[left_index]
            left_index += 1
        else:
            collection[sorted_index] = right_copy[right_index]
            right_index += 1
        sorted_index += 1

    while left_index < len(left_copy):
        collection[sorted_index] = left_copy[left_index]
        left_index += 1
        sorted_index += 1

    while right_index < len(right_copy):
        collection[sorted_index] = right_copy[right_index]
        right_index += 1
        sorted_index += 1


def sleep_sort(collection):
    """Sorts a collection using the sleep sort algorithm.

    Args:
        collection (list): The list of non-negative integers to be sorted.
    Returns:
        list: The sorted list.
    """
    import threading
    import time

    sorted_collection = []
    threads = []

    def sleep_and_append(num):
        time.sleep(num)
        sorted_collection.append(num)

    for num in collection:
        thread = threading.Thread(target=sleep_and_append, args=(num,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return sorted_collection


def strand_sort(collection):
    """Sorts a collection using the strand sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(collection) == 0:
        return collection

    def merge_sorted_lists(list1, list2):
        merged = []
        i = j = 0
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                merged.append(list1[i])
                i += 1
            else:
                merged.append(list2[j])
                j += 1
        merged.extend(list1[i:])
        merged.extend(list2[j:])
        return merged

    result = []
    while collection:
        sublist = [collection.pop(0)]
        i = 0
        while i < len(collection):
            if collection[i] >= sublist[-1]:
                sublist.append(collection.pop(i))
            else:
                i += 1
        result = merge_sorted_lists(result, sublist)

    return result


def cycle_sort(collection):
    """Sorts a collection in place using the cycle sort algorithm.

    Args:
        collection (list): The list to be sorted.
    Returns:
        list: The sorted list.
    """
    n = len(collection)
    for cycle_start in range(n - 1):
        item = collection[cycle_start]
        pos = cycle_start

        for i in range(cycle_start + 1, n):
            if collection[i] < item:
                pos += 1

        if pos == cycle_start:
            continue

        while item == collection[pos]:
            pos += 1

        if pos != cycle_start:
            collection[pos], item = item, collection[pos]

        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if collection[i] < item:
                    pos += 1

            while item == collection[pos]:
                pos += 1

            if item != collection[pos]:
                collection[pos], item = item, collection[pos]

    return collection


def pigeonhole_sort(collection):
    """Sorts a collection using the pigeonhole sort algorithm.

    Args:
        collection (list): The list of integers to be sorted.
    Returns:
        list: The sorted list.
    """
    if len(collection) == 0:
        return collection

    min_value = min(collection)
    max_value = max(collection)
    size = max_value - min_value + 1
    holes = [[] for _ in range(size)]

    for num in collection:
        holes[num - min_value].append(num)

    sorted_collection = []
    for hole in holes:
        sorted_collection.extend(hole)

    return sorted_collection


def stooge_sort(collection, l=0, h=None):
    """Sorts a collection in place using the stooge sort algorithm.

    Args:
        collection (list): The list to be sorted.
        l (int): The starting index of the segment to be sorted.
        h (int): The ending index of the segment to be sorted.
    Returns:
        list: The sorted list.
    """
    if h is None:
        h = len(collection) - 1

    if l >= h:
        return collection

    if collection[l] > collection[h]:
        collection[l], collection[h] = collection[h], collection[l]

    if h - l + 1 > 2:
        t = (h - l + 1) // 3
        stooge_sort(collection, l, h - t)
        stooge_sort(collection, l + t, h)
        stooge_sort(collection, l, h - t)

    return collection


def intro_sort(collection, maxdepth=None):
    """Sorts a collection in place using the introspective sort algorithm.

    Args:
        collection (list): The list to be sorted.
        maxdepth (int): The maximum depth for switching to heapsort.
    Returns:
        list: The sorted list.
    """
    import math

    if maxdepth is None:
        maxdepth = int(math.log2(len(collection))) * 2

    def introsort_helper(arr, start, end, depth_limit):
        size = end - start + 1
        if size <= 16:
            insertion_range(arr, start, end)
        elif depth_limit == 0:
            heap_sort_range(arr, start, end)
        else:
            pivot_index = partition(arr, start, end)
            introsort_helper(arr, start, pivot_index - 1, depth_limit - 1)
            introsort_helper(arr, pivot_index + 1, end, depth_limit - 1)

    def heap_sort_range(arr, start, end):
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1 - start
            right = 2 * i + 2 - start

            if left < n and arr[start + left] > arr[start + largest]:
                largest = left

            if right < n and arr[start + right] > arr[start + largest]:
                largest = right

            if largest != i:
                arr[start + i], arr[start + largest] = (
                    arr[start + largest],
                    arr[start + i],
                )
                heapify(arr, n, largest)

        n = end - start + 1
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[start + i], arr[start] = arr[start], arr[start + i]
            heapify(arr, i, 0)

    introsort_helper(collection, 0, len(collection) - 1, maxdepth)
    return collection
