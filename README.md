---
title: ADMET Triage Panel
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: hERG, Ames and BBB triage with guardrails
---

# ADMET triage panel (Hackathon Track B)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lai-wohjon/admet-triage-panel/blob/main/Launch_App.ipynb)


Predicts three ADMET liabilities for a candidate molecule, with guardrails:

| Endpoint | Meaning of label 1 |
|---|---|
| **hERG** | blocks the hERG channel (cardiotoxicity risk) |
| **AMES** | mutagenic in the Ames test |
| **BBB** | crosses the blood-brain barrier |

Every prediction carries a calibrated probability and a verdict: `HIGH RISK` / `LOW RISK` (or `BBB+` / `BBB-`), `UNCERTAIN` (probability 0.40 to 0.60), or `OUT OF DOMAIN` (nearest training molecule has Tanimoto similarity < 0.30). Invalid or unsupported input (empty, unparsable, inorganic, metals, tiny, huge) is rejected with a reason.

> Research prototype. `LOW RISK` means "no alarm was raised", not "safe". Not for clinical decisions.

## Run the app

**In Colab (no setup):** click the badge above and run the cells. It clones this repo, installs the pinned packages and prints a public link.

**Locally:**
```bash
git clone https://github.com/lai-wohjon/admet-triage-panel.git
cd admet-triage-panel
pip install -r requirements.txt
python app.py        # open the printed local URL, paste one SMILES per line
```

The trained models (`admet_bundle.joblib`, ~87 MB) are **not stored in git**. On first start `app.py` downloads them from the [v1.0 release](https://github.com/lai-wohjon/admet-triage-panel/releases/tag/v1.0) and checks the file's SHA-256 before loading it. If you train your own bundle (below), the local file is used and nothing is downloaded.

> **Security note.** The bundle is a Python pickle, and loading a pickle can execute code. Only load bundles you trained yourself or downloaded from this repository's releases (the hash check enforces the latter).

> **Version note.** The bundle was trained with scikit-learn 1.8.0, which `requirements.txt` pins. Other versions load it with an `InconsistentVersionWarning` and may give different results.

## Repository layout

| File | Purpose |
|---|---|
| `TrackB_ADMET_Hackathon.ipynb` | The full analysis: data, split, models, benchmark, guardrails, error analysis, app |
| `admet_core.py` | Shared standardisation, featurisation, input validation and verdict logic (used by notebook **and** app) |
| `app.py` | Gradio app; downloads the model bundle on first start |
| `Launch_App.ipynb` | Colab launcher for the app |
| `requirements.txt` | Pinned runtime dependencies for the app (also used by the Hugging Face Space) |
| `requirements-train.txt` | Extra packages for re-running the notebook |
| `*.csv` | Benchmark tables, safety metrics, edge-case tests, error log, drug screen (written by the notebook) |

## Reproduce the training

1. `pip install -r requirements-train.txt`
2. Run `TrackB_ADMET_Hackathon.ipynb` top to bottom (CPU is enough; about 25 to 30 minutes on a modern laptop, longer on free Colab). It downloads the three TDC datasets into `data/` (from a Hugging Face mirror of TDC), trains everything and writes `admet_bundle.joblib` and the CSVs. All random seeds are fixed, though tiny (2nd to 3rd decimal) run-to-run differences can occur from multi-threading.
3. `python app.py`

## Method in one paragraph

Datasets are cleaned (largest fragment, neutralisation, canonical SMILES, duplicate and label-conflict removal) and split by Bemis-Murcko scaffold 80/10/10 (randomised, class-balanced, with an assertion that no scaffold crosses partitions). Features are ECFP4 counts, MACCS keys and RDKit 2D descriptors; a GINE graph neural network is trained from scratch for comparison. Models are tuned by scaffold-grouped cross-validation on training data only; the deployed model, thresholds and Platt calibrator are chosen on validation data; the test set is only reported on. Baselines: majority class, Tanimoto kNN, logistic regression. Results use 3 seeds and bootstrap confidence intervals.

## Data and licences

TDC datasets (hERG, AMES, BBB_Martins) via the public `scikit-fingerprints/TDC_*` mirror on Hugging Face; see the TDC site for original sources and terms: https://tdcommons.ai/
