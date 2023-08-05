import sys


def print_memory_usage(max_depth: int = 10) -> None:
    """Print the memory usage by variables
    Args:
        max_depth (int, optional): The max depth of the variables to print. Defaults to 10.
    """

    print("Memory usage:")
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()), key= lambda x: -x[1])[:max_depth]:
        print(f"{name:>30}: {size:>8} bytes")
        