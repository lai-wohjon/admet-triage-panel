"""Gradio MVP for the Track B ADMET triage panel."""
import hashlib, os, urllib.request
from pathlib import Path
import joblib, pandas as pd, gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw
import admet_core as core

# --- model bundle -----------------------------------------------------------------------------
# The trained models (~87 MB) are not stored in git. They are attached to a GitHub Release and
# downloaded once on first start. If you regenerate admet_bundle.joblib by running the training
# notebook, the local file is used as is and nothing is downloaded.
# NOTE: the bundle is a pickle. Only load bundles you trained yourself or downloaded from this repo.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BUNDLE_PATH = HERE / "admet_bundle.joblib"
BUNDLE_URL = os.environ.get(
    "ADMET_BUNDLE_URL",
    "https://github.com/lai-wohjon/admet-triage-panel/releases/download/V1.0/admet_bundle.joblib")
BUNDLE_SHA256 = "e8de89babea4798ab4361593fc5d14df860873081f091e0dbe1bd4c720552818"  # of the v1.0 release file


def load_bundle():
    if not BUNDLE_PATH.exists():
        print(f"Downloading model bundle from {BUNDLE_URL} ...")
        tmp = BUNDLE_PATH.with_suffix(".part")
        urllib.request.urlretrieve(BUNDLE_URL, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != BUNDLE_SHA256:
            tmp.unlink()
            raise RuntimeError(f"Downloaded bundle failed its SHA-256 check (got {digest[:12]}...). Not loading it.")
        tmp.replace(BUNDLE_PATH)
    return joblib.load(BUNDLE_PATH)


BUNDLE = load_bundle()

EXAMPLES = [["CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1\nCn1cnc2c1c(=O)n(C)c(=O)n2C\nCC(=O)Oc1ccccc1C(=O)O"],
            ["Nc1ccc2c(c1)Cc1ccccc12\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"],
            ["[Na+].[Cl-]\nnot a molecule\nCC(=O)Oc1ccccc1C(=O)[O-].[Na+]"]]
ICON = {"HIGH RISK": "🔴", "LOW RISK": "🟢", "UNCERTAIN": "🟡", "OUT OF DOMAIN": "⚪",
        "BBB+ (CNS exposure)": "🔵", "BBB- (peripherally restricted)": "🟤"}

def run(text):
    smiles = [s.strip()[:500] for s in (text or "").splitlines() if s.strip()]
    if len(smiles) > 50:
        gr.Warning("Only the first 50 molecules are processed.")
    smiles = smiles[:50]
    if not smiles:
        return pd.DataFrame({"message": ["Please enter at least one SMILES."]}), []
    res = core.predict_smiles(smiles, BUNDLE)
    table = pd.DataFrame({"SMILES": res["input"], "status": res["status"]})
    for ep in BUNDLE["endpoints"]:
        if f"{ep}_verdict" in res:
            table[ep] = [f"{ICON.get(v, '')} {v} (p={p:.2f})" if isinstance(v, str) else "-"
                         for v, p in zip(res[f"{ep}_verdict"], res[f"{ep}_prob"])]
            table[f"{ep} sim"] = res[f"{ep}_maxsim"].round(2)
    if "canonical" in res:
        table["in training set of"] = [", ".join(ep for ep, a in BUNDLE["endpoints"].items() if isinstance(c, str) and c in a["train_smiles"]) or "-"
                                       for c in res["canonical"]]
    table["notes"] = res["notes"]
    gallery = []
    for s in smiles:
        m, status, _ = core.check_smiles(s)
        if m is not None:
            gallery.append((Draw.MolToImage(m, size=(300, 240)), s[:40]))
    return table, gallery

with gr.Blocks(title="ADMET triage panel") as demo:
    gr.Markdown("# ADMET triage panel\nPredicts **hERG** cardiotoxicity, **Ames** mutagenicity and **blood-brain-barrier** penetration. "
                "Probabilities are calibrated; ⚪ OUT OF DOMAIN means the molecule is too dissimilar from the training data to trust. "
                "hERG probabilities are only approximately calibrated (small validation set). "
                "**Research prototype - not for clinical decisions: LOW RISK means no alarm was raised, not that a compound is safe.**")
    box = gr.Textbox(lines=6, label="SMILES (one per line, max 50)")
    btn = gr.Button("Predict", variant="primary")
    out = gr.Dataframe(label="Results"); gal = gr.Gallery(label="Structures", columns=4)
    btn.click(run, box, [out, gal]); gr.Examples(EXAMPLES, box)

if __name__ == "__main__":
    demo.launch()
