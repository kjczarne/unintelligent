from typing import Iterable, Iterator, Any, List
from itertools import islice


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
