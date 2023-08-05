from .types import as_str_seq, StrSeqOrRecordOrSeq, TokenProbabilityTable
from scipy.stats import entropy
from typing import Dict


def entropy_of_sequence(seq: StrSeqOrRecordOrSeq,
                        probability_table: TokenProbabilityTable):
    """Calculates Shannon Entropy for a given sequence.

    Args:
        seq (StrSeqOrRecordOrSeq): input sequence
        probability_table (TokenProbabilityTable): probability distribution of tokens

    Returns:
        int: Shannon Entropy
    """
    probs = []
    for n in seq:
        prob = probability_table[n.upper()]
        probs.append(prob)
    return entropy(probs)

def expected_entropy(probability_table: TokenProbabilityTable): return entropy(list(probability_table.values()))
expected_entropy.__doc__ = """
Calculates the expected entropy given a discrete probability distribution in the form of a `TokenProbabilityTable`
"""
