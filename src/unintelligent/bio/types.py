from Bio.SeqIO import SeqRecord
from Bio.Seq import Seq
from typing import Dict

StrSeq = str
SeqOrRecord = SeqRecord | Seq
StrSeqOrSeq = StrSeq | Seq
StrSeqOrSeqRecord = StrSeq | SeqRecord
StrSeqOrRecordOrSeq = StrSeq | SeqOrRecord

TokenProbabilityTable = Dict[StrSeq, float]

# pylint: disable-next=multiple-statements,invalid-name
def _unsupported_type_msg(s): return f"Unsupported type f{type(s)}"


# pylint: disable-next=invalid-name
def as_str_seq(s: SeqOrRecord):
    match s:
        case str():
            return s
        case Seq():
            return str(s)
        case SeqRecord():
            return str(s.seq)
        case _:
            raise TypeError(_unsupported_type_msg(s))


# pylint: disable-next=invalid-name
def as_seq(s: StrSeqOrSeqRecord):
    match s:
        case Seq():
            return s
        case str():
            return Seq(s)
        case SeqRecord():
            return Seq(s.seq)
        case _:
            raise TypeError(_unsupported_type_msg(s))


# pylint: disable-next=invalid-name
def as_seq_record(s: StrSeqOrSeq):
    match s:
        case SeqRecord():
            return s
        case Seq():
            return SeqRecord(s)
        case str():
            return SeqRecord(Seq(s))
        case _:
            raise TypeError(_unsupported_type_msg(s))
