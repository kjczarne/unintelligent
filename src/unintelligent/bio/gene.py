import orffinder
from .types import SeqRecord

def get_orfs(seq_record: SeqRecord, min_len: int = 100):
    """Returns all possible open reading frames for a given sequence record

    Returns:
        Dict[str, Any]: a dictionary describing all found ORFs
    """
    # pylint: disable-next=no-member
    return orffinder.getORFs(seq_record, minimum_length=min_len)

from datasets import DatasetDict
from typing import Dict, Any, List
import Bio.Data.CodonTable as CodonTable


def append_gene_to_promoter(seq: str,
                            gene_length: int,
                            is_promoter_seq: bool = True,
                            seed: int = 42,
                            codon_table: CodonTable.CodonTable = CodonTable.standard_dna_table):
    np.random.seed(seed)
    num_codons = gene_length - 6 // 3  # -6 because we account for 1 start and 1 stop codon per gene appended
    actual_gene_length = num_codons * 3  # recover if `gene_length` was not divisible by 3
    if is_promoter_seq:
        # 1. Generate a random in-ORF gene of length `gene_length`:
        start = codon_table.start_codons[0]  # ATG
        stop = np.random.choice(codon_table.stop_codons)  # randomly choose one
        codon_seq = np.random.choice(list(codon_table.forward_table.keys()),
                                     size=num_codons,
                                     replace=True)
        codon_seq = "".join(list(codon_seq))
        # 2. Append the gene to the promoter sequence if the sequence is marked as a promoter:
        return seq + start + codon_seq + stop
    else:
        # 1. Generate a random DNA sequence that does not start with an `ATG` codon
        nucleotides = ["A", "T", "G", "C"]
        rand_nucl_seq = np.random.choice(nucleotides,
                                         size=actual_gene_length + 6,  # +6 because we account for the START/STOP
                                                                       # codons that are missing here
                                         replace=True)
        # 2. Append the padding sequence to the non-promoter sequence
        rand_nucl_seq = "".join(list(rand_nucl_seq))
        return seq + rand_nucl_seq


def append_gene_to_dataset_record(seq: str, label: str, gene_length: int, seed: int = 42):
    match label:
        case 0:
            return dict(sequence=append_gene_to_promoter(seq, gene_length, False, seed), label=label)
        case 1:
            return dict(sequence=append_gene_to_promoter(seq, gene_length, True, seed), label=label)
        case _:
            raise ValueError(f"{label} is not a valid label for a binary classification task")


def append_gene_to_dataset_batch(batch: Dict[str, List[Any]], gene_length: int, seed: int = 42):
    seqs = batch["sequence"]
    labels = batch["label"]
    out = dict(sequence=[], label=[])
    for seq, label in zip(seqs, labels):
        updated_record = append_gene_to_dataset_record(seq, label, gene_length, seed)
        out["sequence"].append(updated_record["sequence"])
        out["label"].append(updated_record["label"])
    return out


def append_genes_to_dataset(dataset_collection: DatasetDict, gene_length: int, seed: int = 42):
    new_dataset_collection = DatasetDict()
    for k in dataset_collection.keys():
        # `batched=True` speeds things up by processing sequences in batches
        new_dataset_collection[k] = dataset_collection[k].map(lambda batch: append_gene_to_dataset_batch(batch,
                                                                                                         gene_length,
                                                                                                         seed),
                                                              batched=True)
    return new_dataset_collection


append_gene_to_promoter(seq, 30, True)
append_gene_to_promoter(seq, 30, False)