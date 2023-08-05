from .types import TokenProbabilityTable

probability_table_dna: TokenProbabilityTable = {
    # https://pubs.acs.org/doi/pdf/10.1021/ja01111a016
    "A": 0.30,  # Chargaff: 0.304
    "T": 0.30,  # Chargaff: 0.301
    "G": 0.19,  # Chargaff: 0.196
    "C": 0.19   # Chargaff: 0.199
}

probability_of_gene: TokenProbabilityTable = {
    # approximately 1.5% of the human genome consists of protein-encoding genes
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9186530/
    "M": 0.015,
    "*": 0.015
}

probability_table_aa: TokenProbabilityTable = {
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7127678/
    "A": 0.0777,
    "C": 0.0157,
    "D": 0.0530,
    "E": 0.0656,
    "F": 0.0405,
    "G": 0.0691,
    "H": 0.0227,
    "I": 0.0591,
    "K": 0.0595,
    "L": 0.0960,
    "M": 0.0238,
    "N": 0.0427,
    "P": 0.0469,
    "Q": 0.0393,
    "R": 0.0526,
    "S": 0.0694,
    "T": 0.0550,
    "V": 0.0667,
    "W": 0.0118,
    "Y": 0.0311,
}
