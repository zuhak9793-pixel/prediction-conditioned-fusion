# Reproducibility

This guide retains the detailed V3.5 protocol. Installation and the main entry points are in [README.md](README.md). Release checks are engineering verification; the included real-data metrics remain historical scientific results.

## Data and split protocol


Obtain the matrix yourself through the [original WS-DREAM project](https://wsdream.github.io/) and [collector page](https://wsdream.github.io/dataset/), following its terms and citation instructions. Use [Zenodo record 1133476](https://zenodo.org/records/1133476), **`wsdream_dataset1.zip`**, MD5 `dde1f1f275d78d65b27015447f9db0e5`, containing `dataset1/rtMatrix.txt`. This 339 × 5825 static matrix is called **Dataset 2** in the original collector documentation; it is not the large dynamic `wsdream_dataset2.zip` archive. No download occurs automatically. Keep data and generated states private.

```powershell
python scripts/prepare_data.py --matrix data/original/dataset1/rtMatrix.txt --out data/prepared
```

Preparation retains finite positive seconds (1,873,838 entries), with no winsorization/log transform. The original seeded recipe creates a common 20% reserved test, per-seed 5% development-validation role, skips the historical 5% development-evaluation segment, and takes nested 1%/5% training subsets from the remaining pool. Seeds are `20261107`, `20261108`, `20261109`; counts are 374,767 test, 93,691 validation and 18,738/93,691 training. Ordering is preserved. The unused development-evaluation labels are not exported.

This script is for newly obtained independent data. Never run it on the original project's sealed data to replace or reinterpret its held-out boundary. Equality with the original private partitions has not been rechecked in this release task.

The collector's old GitHub dataset URL returned 404 during release preparation. The project page and Zenodo metadata were reachable; no dataset download was attempted.

## Frozen experts and public workflow

One example condition (5%, seed 20261107):

```powershell
python -m pcf.cli oof --train data/prepared/20261107_train05.npz --seed 20261107 --out runs/rt05_20261107/oof.npz
python -m pcf.cli fit-gate --oof runs/rt05_20261107/oof.npz --out runs/rt05_20261107/gate.json
python -m pcf.cli train --train data/prepared/20261107_train05.npz --seed 20261107 --out runs/rt05_20261107/models
python -m pcf.cli predict --train data/prepared/20261107_train05.npz --models runs/rt05_20261107/models --gate runs/rt05_20261107/gate.json --coords data/prepared/20261107_validation_coords.npz --out runs/rt05_20261107/validation_predictions.npz
python -m pcf.cli score --role development-validation --targets data/prepared/20261107_validation.npz --predictions runs/rt05_20261107/validation_predictions.npz --out runs/rt05_20261107/validation_metrics.json
```

Repeat with `train01` and the other seeds for six conditions. Output paths must be new. Fitting loads only `role=training`; gate fitting requires `role=training-oof`. Expert and readout identities bind the exact ordered training inputs. Coordinate inference reads only `u/s`, never target members. Input NPZ triples contain `role`, `shape`, `seed`, integer zero-based `u/s`, and positive-second `y` arrays. All inference/scoring coordinates, including unsupported entities, are retained.

NIMF95 uses rank 10, positive-PCC top-10, alpha .4, lambda 30, initial step .01 and seeded MT19937 uniform factors. B3-49 uses rank 8, training mean/population SD, PCG64 normal factors (SD .1), zero unsupported factors, bias penalty .3 and `10 I + 10 (A+I)^-1` factor penalty. OOF experts recompute all statistics using their two training folds; fold IDs use the exact MT19937 permutation/modulo rule. There is no horizon reselection or early stopping.

Each OOF target is omitted from its corresponding expert fit; the readout itself uses all aligned OOF targets. This is not a fully nested independent assessment. Development-validation was historically exposed and supports exploratory analysis.

Predictions include NIMF, B3, equal-weight fusion, learned constant sigmoid, B3-only nonlinear calibration, PCF and adapted DHEM-style K35 confidence weighting. The constant and nonlinear controls use the same training OOF targets and smooth-MAE optimizer as PCF. The constant gate penalizes `a²`; the B3 calibrator penalizes `c0²+c1²`. The DHEM-style rule uses observed per-user training histograms and a half/half fallback for zero total confidence; it is not a full DHEM reproduction.

For the paper's single development ablation, pass `--development-ablation` to `fit-gate`; it additionally fits the unrestricted slope and enables clipped-B3 controls at inference. These are development diagnostics, not extra entries in the main reserved-test table.

For an independent final evaluation, freeze all choices first, use `reserved_test_coords.npz` for `predict`, then explicitly call `score --role reserved-test --acknowledge-reserved-test` with `reserved_test.npz`. Keep new output paths. This instruction does **not** authorize repeating the original Paper5 Final evaluation. Metrics are float64 MAE and square-root mean squared error in seconds; negative estimates and large residuals remain. Seed summaries average the three individual metrics and use sample SD (`ddof=1`), without pooling residuals.

## Independent RA-QoS baseline scope

The optional independent RA-QoS implementation is excluded from this publication copy pending clarification of the supplement copyright provenance. Its two Python modules, frozen configuration, optional dependency file and three optional tests are omitted; no replacement implementation is supplied. The unchanged CLI retains lazy `train-raqos` and `predict-raqos` branches, which are unavailable here. The five core PCF workflow commands and 10 core synthetic tests remain included.

Saved RA-QoS scalar metrics remain unchanged in the public figure/table inputs. They report the historical independent baseline, not an executable baseline supplied by this copy. Its historical selected horizons were 85/103/126 (1%) and 284/349/164 (5%) in seed order. Full baseline reproduction is outside this copy's scope. The supplement MIT notice is retained for attribution history without asserting that its named holder is a paper author; see [PROVENANCE.md](PROVENANCE.md).

## Recreate paper figures and Table I

These verified commands use only included public scalar summaries/diagram geometry:

```powershell
python figures/fig01/plot.py
python figures/fig02/plot.py
python figures/fig03/plot.py
python scripts/reproduce_table.py
```

Figures export PDF/SVG/PNG beside their sources; the table exports Markdown/LaTeX into `build/table/`. Figure 1's editable source is draw.io XML; its local exporter uses Matplotlib's bundled DejaVu Sans, with PDF font subsets and SVG glyph paths. Standalone font files are not bundled; the retained notice is in `LICENSES/DejaVu-fonts.txt`. Figure 2 shows **development-validation** gates and regional errors; Figure 3 and Table I use saved **reserved-test scalar metrics**. Table recreation checks all 48 seed records and 64 rounded numerical fields against the saved aggregates. It does not rerun predictive evaluation. RA-QoS retains the lowest MAE and B3 the lowest RMSE. Seeds share one matrix and overlapping training masks; their SD is not independent-dataset uncertainty.


## Implementation fidelity and historical limits

The released RA-QoS TensorFlow entry point previously failed at its unavailable TensorFlow 1.15 dependency. Its cumulative-loss persistence, input PB subtraction, training-bias penalty and movie configuration differ from the paper specification. These observations do not invalidate the published method. The independent PyTorch implementation does not claim to reproduce the official source or published scores.

Its explicit underspecified choices are thresholds 1, literal zero-fill-minus-PB at all positions, an all-pair training-bias penalty, persistent gamma-weighted loss accumulation and epoch-end adaptation. Widths, regularization and several optimizer settings are disclosed choices, not established published RT defaults. The supplement informs initialization, optimizer settings and the exponential helper, with its MIT notice retained. Adaptive weights saturated in historical runs at native-second scale; finite execution does not establish convergence. The frozen RA-QoS configuration remains in the preserved original candidate and is excluded here.

Original Final evaluation followed a frozen once-only protocol; earlier full-matrix preflight exposure cannot be erased. Historical development-validation was exploratory, OOF meta-assessment was not fully nested independent, and only one dataset was evaluated. Source-release QA establishes no new scientific result or superiority claim.

## Verified scope

The earlier release preparation used an existing Windows Python 3.10.19 environment: native MinGW compilation, all 13 synthetic checks (including optional RA-QoS/CUDA fixtures), bounded synthetic comparisons with selected frozen implementations, three figure renderers and the 64 V3.5 table fields passed. Those were source/interface checks, not full WS-DREAM reproduction.

The final 2026-11-20 review installed the documented core requirements into a separate Windows Python 3.10.19 virtual environment with `include-system-site-packages=false` and no inherited Python package path. MinGW-w64 g++ 8.1.0 compiled the native extension. All 10 core synthetic tests passed; initial discovery skipped three optional RA-QoS tests before PyTorch provisioning. After updating pip from 23.0.1 to 26.3 and installing the documented PyTorch 2.5.1 CUDA 12.4/psutil stack, all 13 synthetic tests passed with no skips, including the tiny CUDA fixture. This does not verify full RA-QoS training. Five separate subprocess CLI stages (OOF, gate fitting, full training, inference and development scoring) passed on 63 synthetic training observations and three unsupported inference coordinates. All three figures rendered and were visually inspected; all 64 Table I fields matched V3.5. Initial sandbox DNS/temp-directory and compiler DLL-launch errors were resolved by approved retries, without scientific-code or dependency-version changes. The Cython `language=` deprecation warning is nonfatal.

No WS-DREAM matrix acquisition/preparation, scientific retraining, six-condition public-interface reproduction or complete portable 500-epoch RA-QoS GPU run was performed in either release check. Original partitions and protected Final artifacts were not reopened, regenerated or hashed. Other operating systems, compilers and font installations remain unverified.

All public figure renderers use DejaVu Sans; the public Figure 1 font was replaced on 2026-11-20 without changing its geometry, scientific text or expressions. The unchanged V3.5 manuscript retains its original figure assets. Font availability and Matplotlib versions can affect appearance and output metadata; bit-identical PDF/SVG output across platforms is not promised. Font terms are separate from Paper5's MIT grant; see [PROVENANCE.md](PROVENANCE.md). The recreated table preserves numerical fields, not the manuscript's full IEEE styling or boldface layout.

The 2026-11-23 publication copy changes licensing/documentation and excludes the five optional RA-QoS files described above. All retained scientific code, core tests, configurations, plots and scalar inputs are byte-identical to the verified 2026-11-20 candidate. The 13-test outcomes above describe that earlier complete candidate; this reduced copy contains 10 core tests.

The `.gitignore` covers documented private/generated paths. `python scripts/check_release.py` deliberately checks the physical source tree, including ignored files; run it on a clean source copy, since a working build contains binaries/caches that it correctly rejects. No generated compiler binaries or font files should be distributed as standalone source assets.
