"""admet_core.py - shared code for the Track B ADMET notebook and the Gradio app.

Keeping standardisation, featurisation, input validation and the traffic-light
verdict in ONE module means training and serving use exactly the same code
(no train/serve skew).
"""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------- constants
MIN_HEAVY, MAX_HEAVY = 3, 150
ALLOWED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}  # H B C N O F Si P S Cl Se Br I
DRUGLIKE_MW = (100.0, 900.0)
AD_SIM_MIN = 0.30          # applicability domain: min Tanimoto to nearest training molecule
UNCERTAIN_BAND = (0.40, 0.60)  # calibrated probabilities in this band -> "UNCERTAIN"

_normalizer = rdMolStandardize.Normalizer()
_uncharger = rdMolStandardize.Uncharger()
_count_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_bit_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

DESC_NAMES = [n for n, _ in Descriptors.descList]
MACCS_NAMES = [f"MACCS_{i}" for i in range(167)]
ECFP_NAMES = [f"ECFP4_{i}" for i in range(2048)]
FEATURE_NAMES = ECFP_NAMES + MACCS_NAMES + DESC_NAMES


# ------------------------------------------------------------ standardisation
def standardize(smiles):
    """SMILES -> cleaned parent RDKit mol (largest fragment, neutralised) or None."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Cleanup(mol)
        mol = _normalizer.normalize(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def check_smiles(smiles):
    """Validate one user-supplied SMILES.

    Returns (mol or None, status, notes). status is 'ok' or 'rejected'.
    Guardrail philosophy: never silently predict on input the model was not built for.
    """
    notes = []
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "rejected", ["Empty input."]
    smiles = smiles.strip()
    raw = Chem.MolFromSmiles(smiles)
    if raw is None:
        return None, "rejected", ["Could not parse SMILES."]
    if len(Chem.GetMolFrags(raw)) > 1:
        notes.append("Multi-component input (salt/mixture): largest fragment used.")
    mol = standardize(smiles)
    if mol is None:
        return None, "rejected", ["Standardisation failed."]
    n_heavy = mol.GetNumHeavyAtoms()
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return None, "rejected", ["Inorganic / no carbon: outside the scope of this model."]
    if n_heavy < MIN_HEAVY:
        return None, "rejected", [f"Too small ({n_heavy} heavy atoms; minimum {MIN_HEAVY})."]
    if n_heavy > MAX_HEAVY:
        return None, "rejected", [f"Too large ({n_heavy} heavy atoms; maximum {MAX_HEAVY})."]
    bad = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in ALLOWED_Z})
    if bad:
        return None, "rejected", [f"Unsupported element(s): {', '.join(bad)}."]
    mw = Descriptors.MolWt(mol)
    if not DRUGLIKE_MW[0] <= mw <= DRUGLIKE_MW[1]:
        notes.append(f"MW {mw:.0f} is outside the drug-like range {DRUGLIKE_MW}: treat with caution.")
    if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
        notes.append("Molecule carries a permanent charge (e.g. quaternary ammonium).")
    return mol, "ok", notes


def scaffold_of(mol):
    """Bemis-Murcko scaffold SMILES ('' for acyclic molecules)."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


# --------------------------------------------------------------- featurisation
def ecfp_counts(mols):
    return np.vstack([_count_gen.GetCountFingerprintAsNumPy(m) for m in mols]).astype(np.float32)


def ecfp_bitvects(mols):
    return [_bit_gen.GetFingerprint(m) for m in mols]


def maccs_bits(mols):
    out = np.zeros((len(mols), 167), dtype=np.float32)
    for i, m in enumerate(mols):
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), out[i])
    return out


def rdkit_descriptors(mols):
    rows = []
    for m in mols:
        r = []
        for _, f in Descriptors.descList:
            try:
                r.append(f(m))
            except Exception:
                r.append(np.nan)
        rows.append(r)
    X = np.asarray(rows, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    X = np.clip(X, -1e6, 1e6)          # tame a handful of absurd values (e.g. Ipc)
    return X.astype(np.float32)


def featurize(mols):
    """Full feature matrix = ECFP4 counts | MACCS | RDKit 2D descriptors."""
    return np.hstack([ecfp_counts(mols), maccs_bits(mols), rdkit_descriptors(mols)])


# -------------------------------------------------------- applicability domain
def max_similarity(fps, train_fps):
    """Tanimoto similarity of each query fingerprint to its nearest training molecule."""
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in fps])


# ------------------------------------------------------------------ calibration
class PlattCalibrator:
    """Sigmoid calibration on the logit of a model's raw probability."""

    def fit(self, p, y):
        z = self._logit(p).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))


# ---------------------------------------------------------------------- verdicts
ENDPOINT_INFO = {
    "hERG":  {"label": "hERG blocker (cardiotoxicity)", "kind": "tox", "pos": "HIGH RISK", "neg": "LOW RISK"},
    "AMES":  {"label": "Ames mutagenic (genotoxicity)", "kind": "tox", "pos": "HIGH RISK", "neg": "LOW RISK"},
    "BBB":   {"label": "Crosses the blood-brain barrier", "kind": "dist", "pos": "BBB+ (CNS exposure)", "neg": "BBB- (peripherally restricted)"},
}


def verdict(endpoint, p_cal, max_sim):
    """Traffic-light verdict combining probability, uncertainty band and applicability domain."""
    info = ENDPOINT_INFO[endpoint]
    if max_sim < AD_SIM_MIN:
        return "OUT OF DOMAIN"
    if UNCERTAIN_BAND[0] < p_cal < UNCERTAIN_BAND[1]:
        return "UNCERTAIN"
    return info["pos"] if p_cal >= 0.5 else info["neg"]


def predict_smiles(smiles_list, bundle):
    """Predict every endpoint in `bundle` for a list of SMILES. Returns a tidy DataFrame."""
    records, mols, keep = [], [], []
    for i, s in enumerate(smiles_list):
        mol, status, notes = check_smiles(s)
        records.append({"input": s, "status": status, "notes": " ".join(notes)})
        if mol is not None:
            mols.append(mol)
            keep.append(i)
    if mols:
        X = featurize(mols)
        fps = ecfp_bitvects(mols)
        canon = [Chem.MolToSmiles(m) for m in mols]
        for ep, art in bundle["endpoints"].items():
            raw = art["model"].predict_proba(X)[:, 1]
            cal = art["calibrator"].predict(raw)
            sim = max_similarity(fps, art["train_fps"])
            for j, i in enumerate(keep):
                records[i]["canonical"] = canon[j]
                records[i][f"{ep}_prob"] = float(cal[j])
                records[i][f"{ep}_maxsim"] = float(sim[j])
                records[i][f"{ep}_verdict"] = verdict(ep, cal[j], sim[j])
    return pd.DataFrame(records)
