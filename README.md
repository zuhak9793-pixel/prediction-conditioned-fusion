# Prediction-Conditioned Fusion (PCF)

Code and public figure inputs for *Prediction-Conditioned Fusion for Sparse Web Service Response-Time Prediction*, manuscript V3.5, by Li Yiming.

PCF combines NIMF and B3 response-time estimates with a two-parameter gate learned from aligned three-fold out-of-fold (OOF) training predictions:

```text
x = log1p(max(b, 0))
w = sigmoid(a - softplus(beta) * x)
prediction = b + w * (n - b)
```

Here `n` and `b` are the NIMF and B3 predictions in seconds. Only the gate feature uses `max`; predictions remain unclipped. The NIMF weight is continuous and nonincreasing in `b`. The objective is mean smooth absolute error (`delta=0.01 s`) plus `1e-4/2 * (a² + softplus(beta)²)`, initialized at `(0, -2)`. Frozen model and optimizer settings are in [configs/frozen.json](configs/frozen.json).

Included implementations are NIMF95, B3-49, equal-weight fusion, constant and nonlinear calibration controls, and adapted DHEM-style confidence weighting. The optional independent RA-QoS implementation is excluded from this publication copy pending supplement provenance clarification; saved RA-QoS scalar results remain for figure/table recreation. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) documents the exact protocol, baseline qualifications and verification limits.

## Installation

Use **Python 3.10** and **MinGW g++ with C++11 support** on Windows. With Python 3.10 selected as `python`, run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
g++ --version
python scripts/build_native.py
python -B -m unittest discover -s tests -v
```

MinGW must be on `PATH`; alternatively, pass `--cxx` and the compiler executable to `build_native.py`. No compiler is downloaded automatically. Build logs go to `build/`. Do not use `python -O`, which disables numerical assertions. Linux, macOS and other compilers remain unverified.

The core requirements pin NumPy 1.26.4, SciPy 1.13.0, Matplotlib 3.8.4 and Cython 3.0.12. This copy includes the 10 core synthetic tests. RA-QoS modules, configuration, optional dependencies and its three optional tests are excluded. The unchanged CLI still lists `train-raqos` and `predict-raqos`; those commands are unavailable in this copy. Use the five core commands documented below.

## Dataset acquisition

Obtain WS-DREAM from the [original project](https://wsdream.github.io/) and [collector documentation](https://wsdream.github.io/dataset/), following their terms and citation instructions. [Zenodo record 1133476](https://zenodo.org/records/1133476) lists **`wsdream_dataset1.zip`**, MD5 `dde1f1f275d78d65b27015447f9db0e5`, containing `dataset1/rtMatrix.txt`.

This is the **339 × 5825 static response-time matrix**, called **Dataset 2** in the collector documentation. It is not the large dynamic `wsdream_dataset2.zip`. Data are not bundled or downloaded automatically; the repository license grants no dataset redistribution rights.

After acquiring and extracting your own copy:

```powershell
python scripts/prepare_data.py --matrix data/original/dataset1/rtMatrix.txt --out data/prepared
```

Keep data, learned states and aligned predictions private. Preparation is for independent reproduction and must not replace the original project's sealed partitions. The split recipe and its verification limits are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#data-and-split-protocol).

## Quick Start

The installation test suite uses synthetic inputs to exercise OOF construction, gate fitting, full-training fits, prediction and development scoring without WS-DREAM. After independent data preparation, the same public CLI runs one 5% condition as follows:

```powershell
python -m pcf.cli oof --train data/prepared/20261107_train05.npz --seed 20261107 --out runs/rt05_20261107/oof.npz
python -m pcf.cli fit-gate --oof runs/rt05_20261107/oof.npz --out runs/rt05_20261107/gate.json
python -m pcf.cli train --train data/prepared/20261107_train05.npz --seed 20261107 --out runs/rt05_20261107/models
python -m pcf.cli predict --train data/prepared/20261107_train05.npz --models runs/rt05_20261107/models --gate runs/rt05_20261107/gate.json --coords data/prepared/20261107_validation_coords.npz --out runs/rt05_20261107/validation_predictions.npz
python -m pcf.cli score --role development-validation --targets data/prepared/20261107_validation.npz --predictions runs/rt05_20261107/validation_predictions.npz --out runs/rt05_20261107/validation_metrics.json
```

Use new output paths. Repeat with `train01` and seeds `20261108` and `20261109` for all six conditions. `python -m pcf.cli --help` lists commands; each subcommand also has `--help`.

Training accepts only `role=training`; gate fitting accepts only `role=training-oof`. Inference reads coordinate members `u/s` without evaluation targets. Scoring retains all coordinates, including unsupported entities, negative predictions and extreme errors. The [reproducibility guide](REPRODUCIBILITY.md) covers independent reserved-test evaluation, input contracts, development controls and RA-QoS.

## Reproducing figures and results

These commands use only included public scalar summaries and diagram geometry:

```powershell
python figures/fig01/plot.py
python figures/fig02/plot.py
python figures/fig03/plot.py
python scripts/reproduce_table.py
```

Figures export PDF/SVG/PNG beside their sources. Figure 1 has editable draw.io geometry and uses DejaVu Sans supplied by Matplotlib; its SVG uses glyph paths and PDF embeds font subsets. The font notice is in [LICENSES/DejaVu-fonts.txt](LICENSES/DejaVu-fonts.txt); standalone font files are not bundled. Table I exports to `build/table/table.md` and `build/table/table.tex`.

Figure 2 uses **development-validation** summaries. Figure 3 and Table I use saved **reserved-test scalar metrics**. Table recreation checks 48 per-seed scalar records and 64 rounded aggregate fields; it does not rerun predictive evaluation. RA-QoS has the lowest saved MAE and B3 the lowest saved RMSE. Three seeds share one matrix and overlapping training masks, so their sample SD is not independent-dataset uncertainty.

Scalar figure/table recreation and synthetic CLI checks are distinct from full model reproduction. Full WS-DREAM retraining, original-partition identity remain unverified in this release review. RA-QoS training is outside this publication copy. See [verified scope](REPRODUCIBILITY.md#verified-scope).

## Citation and licensing

Cite the manuscript as an unpublished work; no accepted venue or DOI is asserted:

```bibtex
@unpublished{li2026pcf,
  author = {Li, Yiming},
  title = {Prediction-Conditioned Fusion for Sparse Web Service Response-Time Prediction},
  year = {2026},
  note = {Manuscript, version 3.5}
}
```

Also credit the original dataset and baseline works when using them. [PROVENANCE.md](PROVENANCE.md) identifies the NIMF source, RA-QoS paper and retained notices.

The author approved the [MIT license](LICENSE) on 2026-11-23 for original Paper5 contributions they have the rights to license, including original code, documentation and figure content. This grant does not assert ownership of upstream material. Third-party source and embedded fonts retain their own terms; see [LICENSES/](LICENSES/) and [PROVENANCE.md](PROVENANCE.md). The project license does not cover WS-DREAM data, separately installed dependencies or compiler runtimes.
