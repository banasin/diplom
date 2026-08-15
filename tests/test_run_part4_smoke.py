import numpy as np
import pandas as pd
from src.run_part4 import build_aptamer_matrix


def test_build_aptamer_matrix_with_and_without_structure():
    pairs = pd.DataFrame({"aptamer_seq": ["ACGTACGT", "GGGGCCCC"],
                          "aptamer_type": ["DNA", "RNA"]})
    X0, names0 = build_aptamer_matrix(pairs, k=2, with_structure=False)
    X1, names1 = build_aptamer_matrix(pairs, k=2, with_structure=True)
    assert X0.shape == (2, 16 + 2)                     # k-меры + is_rna + len
    assert X1.shape == (2, 16 + 2 + 5)                 # + 5 структурных
    assert len(names0) == X0.shape[1] and len(names1) == X1.shape[1]
    assert any(n.startswith("structure_") for n in names1)
