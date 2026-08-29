from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrugRecord:
    drug_id: str
    drug_name: str
    smiles: str


def validate_smiles(smiles: str) -> bool:
    """Validate a SMILES string with RDKit when available, otherwise basic syntax checks.

    Purpose: catch empty or malformed drug structures.
    Inputs: SMILES string scalar. Input shape: ().
    Outputs: boolean validity flag. Output shape: ().
    Data types: str to bool.
    Exceptions: none; invalid strings return False.
    Assumptions: fallback validation is conservative and not chemically authoritative.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return any(ch.isalpha() for ch in smiles) and " " not in smiles


def smiles_to_fingerprint(smiles: str, size: int = 1024, radius: int = 2) -> np.ndarray:
    """Convert SMILES to an ECFP-like 1024-bit fingerprint.

    Purpose: provide RDKit/DeepChem-compatible drug vectors with offline fallback.
    Inputs: SMILES string. Input shape: scalar.
    Outputs: binary fingerprint. Output shape: (size,).
    Data types: str to float32 ndarray.
    Exceptions: ValueError for invalid SMILES.
    Assumptions: fallback hashed n-grams are for demos only, not chemistry evidence.
    """
    if not validate_smiles(smiles):
        raise ValueError(f"Invalid SMILES: {smiles}")
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=size)
        arr = np.zeros((size,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        return arr
    except ImportError:
        arr = np.zeros(size, dtype=np.float32)
        tokens = [smiles[i : i + radius + 1] for i in range(max(1, len(smiles) - radius))]
        for token in tokens:
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % size
            arr[idx] = 1.0
        return arr


def featurize_drugs(drugs: list[DrugRecord], size: int = 1024) -> pd.DataFrame:
    rows = []
    for drug in drugs:
        fp = smiles_to_fingerprint(drug.smiles, size=size)
        rows.append({"drug_id": drug.drug_id, "drug_name": drug.drug_name, "SMILES": drug.smiles, **{f"fp_{i}": fp[i] for i in range(size)}})
    return pd.DataFrame(rows)
