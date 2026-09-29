"""Chemistry helpers matching the competition's connectivity metric."""
from __future__ import annotations


def metric_inchikey14(smiles: str) -> str:
    """Return the tautomer-canonicalized InChIKey first block used by Kaggle.

    RDKit is intentionally imported lazily: the Mac environment may not have
    RDKit installed, while the Kaggle notebook should attach the pinned
    2026.03.3 offline wheel.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem import inchi
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for metric-equivalent molecule deduplication. "
            "Attach the competition's RDKit 2026.03.3 offline wheel in Kaggle."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    key = inchi.MolToInchiKey(mol)
    return key.split("-", 1)[0] if key else ""


def standardize_smiles(smiles: str) -> str:
    """Return a canonical isomeric SMILES after RDKit tautomer canonicalization."""
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise RuntimeError("RDKit is required for structure standardization") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
