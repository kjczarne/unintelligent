from typing import Iterable, Iterator, Any, List
from itertools import islice, accumulate


# pylint: disable-next=invalid-name
def take_n(iterable: Iterable[Any], n: int) -> Iterator[List[Any]]:
    """Takes `n` consecutive elements from an iterable

    Args:
        iterable (Iterable[Any]): any iterable
        n (int): how many elements to grab

    Yields:
        Iterator[List[Any]]: a list of elements from the iterable
    """
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, n))
        if chunk is None:
            break
        yield chunk


def iter_compare(actual: Iterable[Any], expected: Iterable[Any]) -> bool:
    """Compares two iterables element-by element. Optimized with
    `map` and `accumulate` for relatively fast execution times.

    Args:
        actual (Iterable[Any]): first iterable to compare (e.g. a list)
        expected (Iterable[Any]): second iterable to compare (e.g. another list)

    Returns:
        bool: `True` if the two iterables consist of identical elements, `False` otherwise
    """
    return all(accumulate(map(lambda x: x[0] == x[1],  # compare (x[0], x[1]) in the zipped iterables
                              zip(actual, expected, strict=True)),
                          lambda x, y: x and y))
