from .types import as_str_seq, StrSeqOrRecordOrSeq, TokenProbabilityTable
from scipy.stats import entropy


def entropy_of_sequence(seq: StrSeqOrRecordOrSeq,
                        probability_table: TokenProbabilityTable):
    probs = []
    for n in seq:
        prob = probability_table[n.upper()]
        probs.append(prob)
    return entropy(probs)
