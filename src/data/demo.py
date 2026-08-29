from __future__ import annotations

import numpy as np
import pandas as pd

from src.chemistry.fingerprints import DrugRecord
from src.utils.reproducibility import RANDOM_SEED

DEMO_DRUGS = [
    DrugRecord("D001", "Imatinib", "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"),
    DrugRecord("D002", "Erlotinib", "COCCOC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC"),
    DrugRecord("D003", "Gefitinib", "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"),
    DrugRecord("D004", "Osimertinib", "COC1=C(C=C(C=C1)NC(=O)C=C)NC2=NC=CC(=N2)N(C)C"),
    DrugRecord("D005", "Trametinib", "CC1=C(C(=O)N(N1C2=CC=CC=C2)C3=CC(=NC=C3)NC4=CC=C(C=C4)I)F"),
    DrugRecord("D006", "Vemurafenib", "CCCS(=O)(=O)NC1=CC(=C(C=C1)C2=CNC3=NC=C(C=C23)Cl)F"),
    DrugRecord("D007", "Dabrafenib", "CC(C)(C)C1=NC(=C(S1)C2=NC(=NC=C2)N)C3=C(C=C(C=C3)F)F"),
    DrugRecord("D008", "Crizotinib", "CC(C)OC1=CC=C(C=C1)N2CCN(CC2)C3=NC4=C(C=NN4C=C3)C5=CC=C(C=C5)Cl"),
    DrugRecord("D009", "Olaparib", "C1CC1C(=O)N2CCN(CC2)C(=O)C3=CC=CC4=C3C=NN4"),
    DrugRecord("D010", "Palbociclib", "CC1=C(C(=O)N(C2=NC=NC(=C12)N3CCN(CC3)C)C4CCCC4)C"),
    DrugRecord("D011", "Sunitinib", "CCN(CC)CCNC(=O)C1=C(NC2=C1C=C(C=C2)F)C=C3C(=O)NC(=O)N3"),
    DrugRecord("D012", "Sorafenib", "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F"),
    DrugRecord("D013", "Lapatinib", "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC=CC=C5"),
    DrugRecord("D014", "Rucaparib", "CC1=C(C=C2C(=C1)C(=CN2)C3=CC=C(C=C3)F)N"),
    DrugRecord("D015", "Everolimus", "COC1CC(CC(C1O)OC)C2CC(C(C(O2)C)OC)OC"),
    DrugRecord("D016", "Dasatinib", "CC1=NC(=CC(=N1)NC2=NC=C(S2)C3=CC=CC=C3)N4CCN(CC4)CCO"),
    DrugRecord("D017", "Nilotinib", "CC1=C(C=CC(=C1)NC(=O)C2=CN=CC=C2)C3=CC(=NC=C3)N"),
    DrugRecord("D018", "Ribociclib", "CN(C)C1=NC(=C(C=N1)N2CCN(CC2)C)C3=CC=C(C=C3)C(=O)N"),
    DrugRecord("D019", "Alpelisib", "CC1=CC(=NN1)NC2=NC=C(C=C2)C(=O)N3CCOCC3"),
    DrugRecord("D020", "Selpercatinib", "CC1=CC(=NC=C1)C2=NC3=C(C=NN3C=C2)N4CCN(CC4)C"),
]


def synthetic_expression(n_samples: int = 40, n_genes: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    genes = ["TP53", "EGFR", "KRAS", "BRAF", "PIK3CA"] + [f"GENE{i}" for i in range(n_genes - 5)]
    data = rng.normal(5, 2, size=(n_samples, len(genes)))
    df = pd.DataFrame(data, columns=genes)
    df.insert(0, "sample_id", [f"T{i:03d}" for i in range(n_samples)])
    return df


def synthetic_response(sample_ids: list[str], drug_ids: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for sid in sample_ids:
        for did in drug_ids:
            rows.append({"sample_id": sid, "drug_id": did, "response": float(rng.normal(6.0, 1.0))})
    return pd.DataFrame(rows)
