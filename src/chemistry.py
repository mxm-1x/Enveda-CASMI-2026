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
        from rdkit import rdBase
        from rdkit.Chem.MolStandardize import rdMolStandardize
        from rdkit.Chem import inchi
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for metric-equivalent molecule deduplication. "
            "Attach the competition's RDKit 2026.03.3 offline wheel in Kaggle."
        ) from exc

    # The competition's tautomer canonicalizer intentionally caps pathological
    # enumeration. RDKit emits one warning per capped/unkekulizable structure;
    # these are expected in this dataset and would otherwise flood notebook logs.
    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    key = inchi.MolToInchiKey(mol)
    return key.split("-", 1)[0] if key else ""


def candidate_structure(smiles: str) -> tuple[str, str, str, float] | None:
    """Return metric key, standardized SMILES, formula, and exact neutral mass.

    This applies the same pinned tautomer canonicalization used by scoring and
    derives the candidate-table fields from that canonical molecule in one pass.
    """
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Descriptors, inchi, rdMolDescriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as exc:
        raise RuntimeError("RDKit is required to prepare the candidate database") from exc

    rdBase.DisableLog("rdApp.warning")
    rdBase.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    full_key = inchi.MolToInchiKey(mol)
    metric_key = full_key.split("-", 1)[0] if full_key else ""
    if not metric_key:
        return None
    standardized_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    exact_mass = float(Descriptors.ExactMolWt(mol))
    return metric_key, standardized_smiles, formula, exact_mass


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
