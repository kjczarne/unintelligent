"""This module contains a bunch of functions that make it a bit simpler
to work with genetic sequences (DNA, RNA, genes, etc.)
"""

import orffinder
from .types import SeqRecord

def get_orfs(seq_record: SeqRecord, min_len: int = 100):
    """Returns all possible open reading frames for a given sequence record

    Returns:
        Dict[str, Any]: a dictionary describing all found ORFs
    """
    # pylint: disable-next=no-member
    return orffinder.getORFs(seq_record, minimum_length=min_len)
