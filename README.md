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

Note: Some of the code and the organization of some of these information is done with the help of Claude. However, the main ideas are ours. 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lai-wohjon/admet-triage-panel/blob/main/Launch_App.ipynb)

> **New here? (judges, reviewers, general public)** This is a hackathon project (Track B, Cheminformatics and Early Drug Discovery) built by AUA Group 3 for the AUA Joint Lecture Series, Faculty of Pharmacy UI 2026, organised by the Faculty of Pharmacy, Universitas Indonesia. It predicts, from a molecule's structure alone, three early drug-safety risks, and it is designed to say "I don't know" rather than guess.
> - **1. Run the app (no install):** [`Launch_App.ipynb`](Launch_App.ipynb), or click the Colab badge above. Details in [Quick start](#quick-start).
> - **2. See everything behind the model:** [`TrackB_ADMET_Hackathon.ipynb`](TrackB_ADMET_Hackathon.ipynb) holds the data, scaffold split, model training and comparison, SHAP interpretability, calibration, guardrails and error analysis.
> - **3. Read the summary:** [How it works](#how-it-works) and [Results and limitations](#results-and-limitations).
> - **Check the evidence:** [Sources of truth](#sources-of-truth) lists the raw data, the model file with its checksum, and the result tables.
> - **If you are keen to find out more, please contact [laiwohjon@gmail.com](mailto:laiwohjon@gmail.com).**

## Team (AUA Group 3)

| Name | Affiliation |
|---|---|
| Najla Desvaa Yardi | Universitas Indonesia |
| Nayun Kim | Seoul National University |
| Mitayani Wahyu Murti | Universitas Indonesia |
| Lai Woh Jon | National University of Singapore |

A small web app that takes molecules as SMILES strings and flags three ADMET liabilities that commonly kill drug candidates late: **hERG channel block** (cardiac risk), **Ames mutagenicity** (genotoxicity) and **blood-brain-barrier (BBB) penetration** (CNS exposure). Every answer comes with a calibrated probability and a traffic-light verdict, and the app says **"I don't know"** for molecules that look unlike the data it learned from.

> **Research prototype. `LOW RISK` means "no alarm was raised", not "safe". Not for clinical decisions.** It is a triage tool for ranking and prioritising compounds, not a replacement for the assay.

**Contents:** [Team](#team-aua-group-3) · [What it predicts](#what-it-predicts) · [Quick start](#quick-start) · [Using the app](#using-the-app) · [How it works](#how-it-works) · [Results and limitations](#results-and-limitations) · [Repository layout](#repository-layout) · [Reproduce the training](#reproduce-the-training) · [Model file and hosting](#model-file-and-hosting) · [Sources of truth](#sources-of-truth) · [Data and licences](#data-and-licences)

---

## What it predicts

Each endpoint is a separate binary classifier trained on a public [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) dataset ([Huang et al. 2021](https://arxiv.org/abs/2102.09548)). The exact files used are listed under [Sources of truth](#sources-of-truth). "Label 1" is the thing you usually want to avoid, except for BBB, where it depends on your target.

| Endpoint | Label 1 means | Why it matters | Chemistry behind it |
|---|---|---|---|
| **[hERG](https://en.wikipedia.org/wiki/HERG)** | Blocks the hERG potassium channel | Blocking it prolongs the [QT interval](https://en.wikipedia.org/wiki/Long_QT_syndrome) and can cause fatal arrhythmia ([terfenadine](https://en.wikipedia.org/wiki/Terfenadine) and [cisapride](https://en.wikipedia.org/wiki/Cisapride) were withdrawn for this). Regulators screen for it under [ICH S7B](https://www.ich.org/page/safety-guidelines) | Lipophilic, basic amines with aromatic rings are the classic hERG pharmacophore. In the notebook's SHAP analysis the model leans on lipophilicity (MolLogP) and molecular size |
| **AMES** | Mutagenic in the [Ames test](https://en.wikipedia.org/wiki/Ames_test) | Genotoxic compounds are usually stopped immediately (impurity limits follow [ICH M7](https://www.ich.org/page/multidisciplinary-guidelines)) | Structural alerts: aromatic nitro and amine groups, polycyclic aromatics, which are what ECFP substructure counts can express |
| **BBB** | Crosses the [blood-brain barrier](https://en.wikipedia.org/wiki/Blood%E2%80%93brain_barrier) | Wanted for CNS drugs, unwanted for everything else | [Polar surface area](https://en.wikipedia.org/wiki/Polar_surface_area) (TPSA; [Ertl et al. 2000](https://pubmed.ncbi.nlm.nih.gov/?term=Fast+calculation+of+molecular+polar+surface+area+as+a+sum+of+fragment-based+contributions+and+its+application+to+the+prediction+of+drug+transport+properties)), heteroatom count and H-bond donors, the usual CNS rules of thumb. The notebook's SHAP analysis shows exactly these features dominating |

Dataset sizes after cleaning: hERG 607 molecules, Ames 7,255, BBB 1,955. hERG is small, so its numbers carry much more uncertainty than the other two (see [Results and limitations](#results-and-limitations)).

---

## Quick start

There are two ways to run the app. Colab needs nothing installed. Running locally is better if you want to keep using it.

### Option A: Google Colab (no installation)

1. Click the **Open in Colab** badge at the top of this page. A notebook called `Launch_App.ipynb` opens.
2. Run the cells **from top to bottom** using the play button on each cell (or Runtime → Run all).
   - **Block 1** clones this repository and installs the packages. This takes about a minute. If Colab says it needs to restart the session, choose Runtime → Restart session and run Block 1 again.
   - **Block 2** starts the app. The **first start downloads the trained models (about 87 MB) and checks them**, so it may take a minute with no output.
3. When Block 2 finishes it prints two links, and the app may also appear inline under the cell. **For the full-size app, click the public link ending in `.gradio.live`.** The `127.0.0.1` link only exists inside Colab's machine and will not open on yours.
4. The app opens in a new tab. The link works only while the notebook is running and eventually expires. Closing the notebook stops the app.

### Option B: Run it on your own computer

You need **Python 3.12 or 3.13** (the versions this was tested on) and **git**.

```bash
# 1. Get the code
git clone https://github.com/lai-wohjon/admet-triage-panel.git
cd admet-triage-panel

# 2. Create an isolated environment (recommended, keeps the pinned versions from clashing with your other projects)
python -m venv .venv

# 3. Activate it
#    Windows PowerShell:   .venv\Scripts\Activate.ps1
#    Windows cmd:          .venv\Scripts\activate.bat
#    macOS / Linux:        source .venv/bin/activate

# 4. Install the pinned packages (a few minutes; RDKit and Gradio are the big ones)
pip install -r requirements.txt

# 5. Start the app
python app.py
```

On Windows, if `python` is not found, try `py -3.12` instead. If PowerShell refuses to run the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` first, or use the cmd version.

**What you should see, and what to do:**

1. **First run only:** the terminal prints `Downloading model bundle from ...` and then **appears to hang for a minute or more**. It is downloading an 87 MB file and prints no progress. Wait. Later runs skip this step.
2. Then it prints `* Running on local URL:  http://127.0.0.1:7860`.
3. **The browser does not open by itself.** Copy that address into your browser, or Ctrl+click it in most terminals. If port 7860 is busy, Gradio picks the next free one, so use whatever address is printed.
4. Keep the terminal open while you use the app. Press **Ctrl+C** in the terminal to stop it.

### If something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| Nothing opens in the browser | Expected. Open the `http://127.0.0.1:...` address printed in the terminal by hand. |
| Stuck after `Downloading model bundle...` | The first-run download is silent. Give it a couple of minutes. If it errors, download `admet_bundle.joblib` yourself from the [Releases page](https://github.com/lai-wohjon/admet-triage-panel/releases/tag/V1.0) and put it **next to `app.py`**. The app then skips the download. |
| `failed its SHA-256 check` | The download was corrupted or is not the official file. Delete any partial file and run again, or use the manual download above. |
| `InconsistentVersionWarning` mentioning scikit-learn | The wrong scikit-learn is installed. The models were trained with 1.8.0: `pip install scikit-learn==1.8.0`. |
| `ModuleNotFoundError` (rdkit, gradio, ...) | The virtual environment is not activated, or step 4 was skipped. Activate it and rerun `pip install -r requirements.txt`. |
| Colab link opens but shows an error | Re-run Block 2. If the problem remains, restart the Colab session and run Blocks 1 and 2 again. |

---

## Using the app

### Input

Type or paste **one [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) per line** into the box (maximum 50 molecules per run, 500 characters each) and press **Predict**. Three example sets are provided under the box; click one to fill it in.

Where to get SMILES: the "Canonical SMILES" field on a [PubChem](https://pubchem.ncbi.nlm.nih.gov/) compound page, "Copy as SMILES" in ChemDraw or MarvinSketch, or `Chem.MolToSmiles(mol)` in RDKit. Salts and mixtures are accepted: the largest fragment is used and a note tells you so.

### Reading the results

You get a table (one row per input) and a gallery of the drawn structures.

| Column | Meaning |
|---|---|
| `status` | `ok`, or `rejected` with the reason in `notes` |
| `hERG`, `AMES`, `BBB` | Verdict, plus `p=`, the **calibrated probability of label 1** (blocker / mutagenic / crosses the BBB) |
| `hERG sim`, `AMES sim`, `BBB sim` | [Tanimoto similarity](https://doi.org/10.1186/s13321-015-0069-3) ([ECFP4](https://pubmed.ncbi.nlm.nih.gov/20426451/)) to the **single most similar training molecule** for that endpoint. 1.0 means an identical structure is in the training data |
| `in training set of` | Endpoints whose training data already contain this exact molecule. Agreement there is memory, not prediction |
| `notes` | Warnings such as salt stripped, unusual molecular weight, permanent charge |

**Verdicts**

| Icon | Verdict | When |
|---|---|---|
| 🔴 | `HIGH RISK` | hERG or Ames probability ≥ 0.60 |
| 🟢 | `LOW RISK` | hERG or Ames probability ≤ 0.40. Not a safety clearance |
| 🔵 / 🟤 | `BBB+ (CNS exposure)` / `BBB- (peripherally restricted)` | BBB probability ≥ 0.60 / ≤ 0.40 |
| 🟡 | `UNCERTAIN` | Probability between 0.40 and 0.60: the model is close to a coin flip |
| ⚪ | `OUT OF DOMAIN` | Similarity to the nearest training molecule is below 0.30. **Ignore the probability shown**; the model has not seen anything like this |

**There is no "moderate risk" category.** The middle zone is called `UNCERTAIN`. A molecule the app refuses to score (`status` = `rejected`) gets no verdict at all, only a reason in `notes`.

**Order of checks (the first one that applies wins):**

1. **Is the input acceptable?** If not, it is rejected.
2. **Is the molecule similar enough to the training data?** If the closest training molecule is below 0.30 similar, the answer is `OUT OF DOMAIN`. This is checked separately for each endpoint, because each has its own training set, so one molecule can be out of domain for hERG yet still get an Ames answer.
3. **Is `p` strictly between 0.40 and 0.60?** Then `UNCERTAIN`.
4. **Otherwise:** `p` of 0.60 or more gives `HIGH RISK` (or `BBB+`), and `p` of 0.40 or less gives `LOW RISK` (or `BBB-`).

### Worked examples

These come from the released models (v1.0), so you can check your install against them.

| Molecule (SMILES) | hERG | Ames | BBB |
|---|---|---|---|
| [Terfenadine](https://en.wikipedia.org/wiki/Terfenadine) `CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1` | 🔴 HIGH RISK (p=0.96) | 🟢 LOW RISK (p=0.09) | 🟤 BBB- (p=0.21) |
| [Benzo[a]pyrene](https://en.wikipedia.org/wiki/Benzo%28a%29pyrene) `c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34` | ⚪ OUT OF DOMAIN | 🔴 HIGH RISK (p=1.00) | ⚪ OUT OF DOMAIN |
| [Fexofenadine](https://en.wikipedia.org/wiki/Fexofenadine) `CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1` | 🟡 UNCERTAIN (p=0.51) | 🟢 LOW RISK (p=0.09) | 🟤 BBB- (p=0.11) |
| Aspirin `CC(=O)Oc1ccccc1C(=O)O` | 🟢 LOW RISK (p=0.01), sim 0.31 | 🟢 LOW RISK (p=0.01), in Ames training set | 🟡 UNCERTAIN (p=0.51) |

How to read them: terfenadine is the textbook hERG blocker, a lipophilic basic amine, and is flagged. Fexofenadine, terfenadine's safer acid cousin, comes out `UNCERTAIN` instead of a confident answer either way. Benzo[a]pyrene is a classic mutagen and is flagged for Ames, while the hERG and BBB columns correctly refuse to answer because it is a flat, fully aromatic hydrocarbon unlike anything in those training sets. Aspirin's hERG similarity of 0.31 is only just above the 0.30 cutoff, a reminder that "in domain" is a threshold, not a guarantee.

### Rejected input

The app refuses, with a reason, rather than guessing on input it was not built for: empty or unparsable SMILES, molecules with no carbon (water, salts, cisplatin), fewer than 3 or more than 150 heavy atoms, and elements other than H, B, C, N, O, F, Si, P, S, Cl, Se, Br, I. Molecular weights outside 100 to 900 and permanently charged molecules are accepted with a warning in `notes`.

---

## How it works

This section explains what each step does **in this project specifically**. The full code and the evidence for every claim is in `TrackB_ADMET_Hackathon.ipynb`.

### 1. Cleaning

Each SMILES is parsed with RDKit, reduced to its **largest fragment** (strips counter-ions), neutralised, and converted to a canonical SMILES. Duplicates are merged. **If the same structure appears with conflicting labels it is dropped**, because there is no right answer to learn and whichever copy lands in the test set would randomly flatter or punish the model. The same cleaning code (`admet_core.py`) runs at training time and inside the app, so there is no train/serve mismatch.

### 2. Scaffold split (the key evaluation decision)

Medicinal chemists make series of close analogues, so a random split puts siblings in both train and test and the model scores well by remembering a twin (a well-documented problem: [Wallach & Heifets 2018](https://arxiv.org/abs/1706.06619)). To mimic the real job (scoring chemistry the model has not seen), molecules are grouped by **[Bemis-Murcko scaffold](https://pubmed.ncbi.nlm.nih.gov/8709122/)** (the ring systems and linkers with side chains stripped) and **whole scaffold groups** go to train, validation or test in an 80/10/10 split. Code asserts that no scaffold appears in two partitions. Acyclic molecules share one empty-scaffold group, which stays together.

A plain deterministic scaffold split (biggest groups to train, singletons to test) produced test sets with badly skewed class balance here, so the split is randomised and the best of 300 shuffles is kept for class balance. In the notebook's own comparison, a random split inflated the Ames ROC-AUC by about 0.06 (0.92 against 0.86). Note that the hERG numbers cannot be told apart between the two splits because its test set is so small.

### 3. Features

Each molecule becomes a vector of **2,432 numbers**:

- **2,048 ECFP4 counts** ([Morgan / extended-connectivity fingerprint](https://pubmed.ncbi.nlm.nih.gov/20426451/), radius 2): how many times each hashed circular substructure occurs. This is what lets the model see things like a basic amine, a nitroaromatic or a fused polycyclic.
- **167 [MACCS keys](https://pubmed.ncbi.nlm.nih.gov/?term=Reoptimization+of+MDL+keys+for+use+in+drug+discovery)**: predefined substructure flags.
- **217 [RDKit](https://www.rdkit.org/) 2D descriptors**: physicochemical quantities such as logP, TPSA, molecular weight, H-bond donors and acceptors, which are the quantities the BBB and hERG rules of thumb are written in.

### 4. Models: a ladder where each rung must earn its place

| Model | How it is used here | Deployed? |
|---|---|---|
| Majority class | Predicts the base rate. Floor (AUC 0.5) | No |
| **kNN-Tanimoto** | For a query, find the 5 most similar training molecules by ECFP4 Tanimoto similarity and take a similarity-weighted vote of their labels. This is the "similar molecules behave alike" assumption of QSAR in its barest form. It sets the bar a fancier model must clear. | No (baseline) |
| Logistic regression | Linear model on log-scaled ECFP4 counts, regularisation tuned by scaffold-grouped cross-validation | No (baseline) |
| **[Random Forest](https://doi.org/10.1023/A:1010933404324)** | 500 trees on all 2,432 features. Small, CPU-only and explainable with [SHAP](https://arxiv.org/abs/1705.07874) | **Yes, all three endpoints** |
| [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) | Gradient boosting, tuned by scaffold-grouped cross-validation | No (compared) |
| GNN ([GINE](https://arxiv.org/abs/1905.12265)) | Graph neural network written from scratch in PyTorch: atoms are nodes, bonds are edges, 3 message-passing layers | No (compared) |
| Ensemble | Combines the predictions of the advanced models (details in the notebook) | No (compared) |

The **shipped model per endpoint was chosen on validation AUC, never on the test set**. Random Forest won all three. The GNN did not beat it on any endpoint.

**How nearest-neighbour similarity is used twice.** kNN-Tanimoto is the baseline model above. The same Tanimoto machinery also powers the app's **applicability domain** check (step 6), but there it is not used to predict anything: it only measures how close the nearest training molecule is.

### 5. Calibration

Tree ensembles usually do not output well-calibrated probabilities (see the [scikit-learn guide to calibration](https://scikit-learn.org/stable/modules/calibration.html)). A **[Platt calibrator](https://doi.org/10.1145/1102351.1102430)** (a logistic regression on the model's logit) is fitted on the **validation set only**, so that "0.8" should mean roughly "80% of molecules scored like this really are positive". It improved Ames ([Brier score](https://en.wikipedia.org/wiki/Brier_score) 0.140 to 0.134), was neutral for BBB, and made hERG slightly worse, because only about 60 hERG molecules were available to fit it. **Treat hERG probabilities as rankings with rough meaning, not as exact risks.**

### 6. Guardrails

The verdict logic (`admet_core.verdict`) combines three safeguards:

1. **Input validation**: refuse what the model was not built for (see [Rejected input](#rejected-input)).
2. **Applicability domain**: if the nearest training molecule has Tanimoto similarity below **0.30**, answer `OUT OF DOMAIN` (the idea that prediction accuracy falls with distance from the training set is shown in [Sheridan et al. 2004](https://pubmed.ncbi.nlm.nih.gov/?term=Similarity+to+molecules+in+the+training+set+is+a+good+discriminator+for+prediction+accuracy+in+QSAR)). In the notebook's subgroup analysis (ensemble model), ROC-AUC on the least similar third of the test set was 0.77 (hERG), 0.77 (Ames) and 0.85 (BBB), against 0.94 to 1.00 for the most similar third.
3. **Uncertainty band**: calibrated probabilities between **0.40 and 0.60** give `UNCERTAIN` instead of a coin-flip verdict.

Both thresholds were fixed in advance, not tuned on the test set.


> **More interpretability and calibration detail is in the notebook.** This README gives the short version. [`TrackB_ADMET_Hackathon.ipynb`](TrackB_ADMET_Hackathon.ipynb) has the full analysis, with the plots and code:
> - **Model interpretability:** section 6.2, SHAP importance bars and beeswarm plots for hERG and BBB, with a check of whether the model rediscovers known chemistry.
> - **Calibration:** section 7, reliability curves on the test set before and after Platt scaling, with Brier scores.
> - **Knowing what it doesn't know:** section 8, the accuracy-versus-coverage trade-off, what `LOW RISK` really means (sensitivity, negative predictive value and a safety-tuned threshold).
> - **Robustness:** section 9 (edge-case stress tests and subgroup/bias checks) and section 10 (the worst false positives and negatives).
> - **Uncertainty on the benchmark:** section 5.1, bootstrap confidence intervals and paired differences.

---

## Results and limitations

Held-out scaffold test set, deployed Random Forest:

| Endpoint | Test molecules | [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) | Gain over logistic regression (95% CI) |
|---|---|---|---|
| hERG | 62 | 0.909 | +0.004 (−0.054 to +0.057): none detectable |
| Ames | 726 | 0.886 | +0.031 (+0.013 to +0.050): real but modest |
| BBB | 196 | 0.917 | +0.043 (−0.006 to +0.090): borderline |

Read these honestly:

- **Logistic regression on fingerprints is a strong baseline.** The Random Forest clearly beats it only for Ames, where there is the most data. With about 500 training molecules for hERG, extra model complexity buys nothing measurable.
- **hERG results are fragile.** A 62-molecule test set gives wide intervals, and its probability calibration is only approximate.
- **`LOW RISK` is not clearance.** In the Ames test set, 44 of the 223 molecules given a LOW RISK verdict (about 20%) were in fact mutagenic. At the plain 0.5 cut-off the model catches about 86% of hERG blockers, 81% of Ames mutagens and 98% of BBB-positives, so some true positives always slip through.
- **It can be confidently wrong inside its domain**, for example on some polycyclic aromatics for Ames.
- **Scope is narrow**: drug-like organic molecules from public datasets whose labels contain noise. It predicts from 2D structure alone: no 3D conformation, protonation state, metabolism or dose.

The tables behind these numbers are in the repository: `benchmark_summary.csv`, `benchmark_ci.csv`, `benchmark_deployed_vs_baselines.csv`, `safety_metrics.csv`, `edge_case_tests.csv` (14 of 14 input edge cases pass), `error_analysis_log.csv` (the worst false positives and negatives) and `library_screen.csv` (16 well-known drugs, with a flag for which ones the model had already seen in training).

---

## Repository layout

| File | Purpose |
|---|---|
| **[`Launch_App.ipynb`](Launch_App.ipynb)** | **Start here to run the app.** Colab launcher: run two cells, no install |
| **[`TrackB_ADMET_Hackathon.ipynb`](TrackB_ADMET_Hackathon.ipynb)** | **Start here to see how the model was built.** The full analysis: data, split, model training and benchmark, SHAP interpretability, calibration, guardrails, error analysis, and writing the app |
| `README.md` | This page |
| `app.py` | The Gradio app. Downloads the model file on first start |
| `admet_core.py` | Shared cleaning, featurisation, input validation, similarity and verdict logic, used by both the notebook and the app |
| `requirements.txt` | Pinned packages needed to run the app |
| `requirements-train.txt` | Extra packages needed to re-run the notebook |
| `benchmark_*.csv`, `safety_metrics.csv`, `edge_case_tests.csv`, `error_analysis_log.csv`, `library_screen.csv` | Result tables written by the notebook |

**Two versions of the notebook exist.** The one in this repository is a cleaned-up copy: its package-install and file-writing cells were replaced by a step that clones this repository, and one sentence in section 11.3 was corrected. The original, as submitted, also wrote `admet_core.py`, `app.py` and `requirements.txt` itself and installed the newest packages; the team keeps it as the process record. `admet_core.py` here is identical to what the original notebook writes. `app.py` differs only by the model-download block.

Not in the repository (listed in `.gitignore`): `admet_bundle.joblib` (the trained models; see below) and `data/` (the datasets and feature cache, which the notebook downloads and generates).

---

## Reproduce the training

1. Install the full set of packages: `pip install -r requirements-train.txt`.
2. Open `TrackB_ADMET_Hackathon.ipynb` and run it **top to bottom**. A CPU is enough: about 25 to 30 minutes on a modern laptop, longer on free Colab. In Colab, the first cell fetches this repository for you.
3. The notebook downloads the three TDC datasets into `data/` (from a Hugging Face mirror of TDC), trains and benchmarks everything, and writes `admet_bundle.joblib` plus the result CSVs.
4. `python app.py` now uses your freshly trained local `admet_bundle.joblib` and downloads nothing.

All random seeds are fixed, though tiny (2nd to 3rd decimal) run-to-run differences can occur from multi-threading. Your retrained bundle will not match the published SHA-256, which only applies to the downloaded v1.0 file.

---

## Model file and hosting

- **Why the models are not in git:** `admet_bundle.joblib` is about 87 MB, so it is attached to the [V1.0 release](https://github.com/lai-wohjon/admet-triage-panel/releases/tag/V1.0) instead of the repository history. `app.py` downloads it on first start and checks its SHA-256 before loading.
- **Security note:** the bundle is a Python [pickle](https://docs.python.org/3/library/pickle.html), and loading a pickle can execute code. Only load bundles you trained yourself or downloaded from this repository's releases. The hash check enforces the second case for the automatic download. A bundle file already sitting next to `app.py` is loaded **without** a hash check, so only put files there that you trust.
- **Version note:** the bundle was trained with scikit-learn 1.8.0 and RDKit 2026.9.1, which `requirements.txt` pins. Other scikit-learn versions load it with an `InconsistentVersionWarning` and may behave differently. RDKit is pinned because the 217 descriptors come from its descriptor list, and a different RDKit version can change that list so the features no longer match what the model was trained on.
- **Hugging Face Space (optional):** the settings block at the very top of this file (shown as a small table on GitHub) configures this repository to run as a Gradio Space, if you create one and push the repository to it. This has not been tested on Hugging Face itself.

---

## Sources of truth

Everything reported above can be traced to one of these.

**Raw data (not redistributed here; downloaded by the notebook)**

| Endpoint | Original publication | TDC page | File the notebook downloads | Rows before cleaning | Positive rate |
|---|---|---|---|---|---|
| hERG | [Wang et al. 2016](https://pubmed.ncbi.nlm.nih.gov/?term=ADMET+evaluation+in+drug+discovery+16+Predicting+hERG+blockers+by+combining+multiple+pharmacophores+and+machine+learning+approaches) | [TDC toxicity tasks](https://tdcommons.ai/single_pred_tasks/tox/) | [`tdc_herg.csv`](https://huggingface.co/datasets/scikit-fingerprints/TDC_herg/resolve/main/tdc_herg.csv) | 655 | 68.9% |
| AMES | [Xu et al. 2012](https://pubmed.ncbi.nlm.nih.gov/23030379/) | [TDC toxicity tasks](https://tdcommons.ai/single_pred_tasks/tox/) | [`tdc_ames.csv`](https://huggingface.co/datasets/scikit-fingerprints/TDC_ames/resolve/main/tdc_ames.csv) | 7,278 | 54.6% |
| BBB | [Martins et al. 2012](https://pubmed.ncbi.nlm.nih.gov/?term=A+Bayesian+approach+to+in+silico+blood-brain+barrier+penetration+modeling) | [TDC ADME tasks](https://tdcommons.ai/single_pred_tasks/adme/) | [`tdc_bbb_martins.csv`](https://huggingface.co/datasets/scikit-fingerprints/TDC_bbb_martins/resolve/main/tdc_bbb_martins.csv) | 2,030 | 76.4% |

The files are copies of the TDC tables on a public Hugging Face mirror (`scikit-fingerprints/TDC_*`), used because TDC's own download server is sometimes unreachable. The mirror holds a few more rows than the official TDC counts (for example 655 against 648 for hERG); cleaning removes duplicates and conflicting labels, leaving 607, 7,255 and 1,955 molecules. The row counts and positive rates in the table are what the notebook prints when it loads the files, so you can check your download against them. The exact download code is the cell that defines `SOURCES` in section 1.1 of the notebook.

**Trained model**: [`admet_bundle.joblib`](https://github.com/lai-wohjon/admet-triage-panel/releases/tag/V1.0), [SHA-256](https://en.wikipedia.org/wiki/SHA-2) `e8de89babea4798ab4361593fc5d14df860873081f091e0dbe1bd4c720552818`.

**Every number in Results**: the CSV tables in this repository (`benchmark_summary.csv`, `benchmark_ci.csv`, `benchmark_deployed_vs_baselines.csv`, `benchmark_all_seeds.csv`, `safety_metrics.csv`, `edge_case_tests.csv`, `error_analysis_log.csv`, `library_screen.csv`), written by `TrackB_ADMET_Hackathon.ipynb`.

**Methods references**: [Tanimoto similarity](https://doi.org/10.1186/s13321-015-0069-3) · [ECFP fingerprints](https://pubmed.ncbi.nlm.nih.gov/20426451/) · [MACCS keys](https://pubmed.ncbi.nlm.nih.gov/?term=Reoptimization+of+MDL+keys+for+use+in+drug+discovery) · [Bemis-Murcko scaffolds](https://pubmed.ncbi.nlm.nih.gov/8709122/) · [why random splits flatter models](https://arxiv.org/abs/1706.06619) · [Random Forest](https://doi.org/10.1023/A:1010933404324) · [LightGBM](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree) · [GINE graph network](https://arxiv.org/abs/1905.12265) · [probability calibration](https://doi.org/10.1145/1102351.1102430) · [similarity and accuracy in QSAR](https://pubmed.ncbi.nlm.nih.gov/?term=Similarity+to+molecules+in+the+training+set+is+a+good+discriminator+for+prediction+accuracy+in+QSAR) · [SHAP](https://arxiv.org/abs/1705.07874) · [RDKit](https://www.rdkit.org/) · [Gradio](https://www.gradio.app/) · [TDC](https://arxiv.org/abs/2102.09548)

---

## Data and licences

The datasets are the TDC benchmark versions of hERG (Wang et al. 2016), Ames (Xu et al. 2012) and BBB_Martins (Martins et al. 2012), downloaded from the public `scikit-fingerprints/TDC_*` mirror on Hugging Face. They are **not** redistributed in this repository. See the TDC site for the original sources and terms: https://tdcommons.ai/
