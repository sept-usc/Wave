def is_primitive(seq):
    n = len(seq)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            unit = seq[:i]
            if unit * (n // i) == seq:
                return False
    return True


def find_max_repeat_cycle(arr):
    n = len(arr)
    best_pattern = None
    best_count = 1
    best_starting_idx = 0
    for length in range(n // 2, 0, -1):
        for start in range(n - length * 2 + 1):
            starting_idx = start
            pattern = arr[start : start + length]
            if not is_primitive(pattern):
                continue
            count = 1
            idx = start + length
            while idx + length <= n and arr[idx : idx + length] == pattern:
                count += 1
                idx += length
            if count > 1 and (
                best_pattern is None
                or count > best_count
                or (count == best_count and starting_idx > best_starting_idx)
            ):
                best_pattern = pattern
                best_count = count
                best_starting_idx = start
    if best_pattern is None:
        return None, None, 1
    else:
        return best_starting_idx, best_pattern, best_count


if __name__ == "__main__":
    arr = [
        "c",
        "c",
        "c",
        "a",
        "b",
        "a",
        "b",
        "a",
        "b",
        "c",
        "b",
        "c",
        "b",
        "c",
        "c",
        "c",
    ]
    print(find_max_repeat_cycle(arr))
