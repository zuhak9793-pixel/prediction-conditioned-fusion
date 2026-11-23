# Provenance and licensing

Original candidate reviewed 2026-11-20; publication copy prepared 2026-11-23. The scientific specification is manuscript V3.5, *Prediction-Conditioned Fusion for Sparse Web Service Response-Time Prediction*. The author explicitly approved MIT licensing on 2026-11-23 for original Paper5 contributions they have the rights to license. The root MIT text applies to those rights only, including original source, documentation and figure content; it makes no claim of upstream ownership. Upstream source and embedded font notices remain independently applicable. No dataset rights are granted by this repository.

| Included component | Origin and changes |
|---|---|
| `native/` | Project-associated [WS-DREAM](https://github.com/rzhu3/WS-DREAM) revision `78ddcffe10976842f8fcdf3c8d14e968da067df6`, Jamie Zhu / WS-DREAM, MIT. `c_NIMF.cpp/.h` and `c_UIPCC.cpp/.h` are derived upstream C++ files with author/license headers retained. `NIMF.pyx` is the Paper5 compatibility binding (with upstream attribution); `bridge.h` is the Paper5 interface to the attributed upstream mathematics and fixes TopK at 10. All six are the unchanged November 8 compatibility-port bytes: corrected array deletion, owned factors, Cython 3 binding, coordinate inference; objective, gradient, line search and PCC unchanged. |
| `pcf/nimf.py` | Frozen November 15 OOF helper plus the original explicit MT19937 initialization. Function bodies reformatted; unused validation-selection orchestration omitted. |
| `pcf/b3.py` | Paper5 rank-8 training-only B3 from the November 15 paired experiment; fixed 49 sweeps. Unused B0/B1/B2/B4 training paths and scalar-penalty branch removed. B3 arithmetic unchanged. |
| `pcf/fusion.py` | Frozen PCF objective from `qos_fusion_cross_seed_20261115/readouts.py`; constant, B3 nonlinear and unrestricted controls from `qos_minimal_fusion_ablation_20261115/run.py`. Unified interface, exact formula/optimizer settings retained. Historical aggregate method name `conditional` means PCF. |
| `pcf/pcw.py` | Paper5 November 14 adapted DHEM-style histogram/confidence functions. K=35, inclusive upper endpoint, zero-confidence half/half fallback. This is not the complete published DHEM system. |
| RA-QoS implementation (excluded) | The original candidate has an independent paper-based PyTorch implementation with supplement-informed initialization, optimizer and exponential helper. Both Python modules, its config, optional dependencies and tests are omitted from this publication copy pending provenance clarification. No substitute is introduced. |
| Figures/results | Paper5-owned published scalar summaries and vector diagram. Figure 2 uses the V3.1 footer-free source retained by V3.5; Figure 3 uses the November 18 polished source. Extraction/private-project lookup branches removed; render logic retained. Final CSVs contain aggregate metrics and per-seed scalar errors only. |
| New interfaces/tests | Portable role-pure NPZ contracts, CLI, data preparation, identity binding, build/table/safety scripts and synthetic checks. They do not alter numerical updates. |

`source_manifest.json` identifies the exact original source/configuration/aggregate files inspected, with bounded source hashes. These are lineage references, not dependencies on a private checkout. No original data, checkpoint or aligned prediction was hashed for this release task.

NIMF's complete MIT notice is in [LICENSES/WS-DREAM-MIT.txt](LICENSES/WS-DREAM-MIT.txt), and its author headers are retained.

## RA-QoS: academic credit and supplement notice

**Academic citation:** Shun Fu, Junnan Li and Lufeng Wang (2025), *RA-QoS: A robust autoencoder-based QoS predictor for highly accurate web service QoS prediction*, PeerJ Computer Science 11:e2928, [doi:10.7717/peerj-cs.2928](https://doi.org/10.7717/peerj-cs.2928). This credits the published method.

**Supplement copyright:** the recovered supplement separately states `Copyright (c) 2023 Di Wu` in its MIT notice. The complete notice is retained unchanged in [LICENSES/RA-QoS-supplement-MIT.txt](LICENSES/RA-QoS-supplement-MIT.txt) as an attribution record for the source-informed initialization, optimizer and exponential-helper elements in the original candidate. Those two modules are excluded from this publication copy. Academic credit does not replace that notice. Di Wu is not identified here as a paper author; no inspected evidence establishes the relationship between that copyright holder and the three named paper authors. No correction to the holder is inferred. The notice does not establish a Gitee revision or rights in bundled data.

The excluded independent implementation used the explicit choices in its original configuration: thresholds 1, literal zero-fill-minus-PB, all-pair training-bias penalty, persistent weighted losses, epoch-end adaptation, and training-only horizon selection. It is not an official/source-exact or published-score reproduction. No supplement source archive, dataset or checkpoint is distributed.

Dependencies are installed separately, not vendored. Inspected installed package records identify NumPy/SciPy BSD notices, Matplotlib's PSF-based license, PyTorch BSD-3-Clause and psutil BSD-3-Clause; Cython 3.0.12 carries Apache-2.0. Their wheels/compiler runtimes have additional notices that are not relicensed by the root MIT text. Generated MinGW runtime DLLs and binaries are excluded from this source candidate. WS-DREAM code licensing must not be conflated with dataset redistribution terms. The [original collector page](https://wsdream.github.io/dataset/) and [Zenodo record](https://zenodo.org/records/1133476) provide acquisition information; raw data remain excluded. The old GitHub dataset URL returned 404 in this check; current archive download and dataset-rights reconciliation were not attempted.

The release preserves third-party attribution and development provenance. It makes no assertion that the code was produced without AI assistance.

## Ownership and license scope

The approved MIT grant covers only original Paper5 contributions for which the author holds licensing rights. The retained PCF/B3/OOF/PCW orchestration, portable interfaces, scripts, tests, configurations, original figure content and public scalar summaries fall within the author-approved contribution scope. The NIMF orchestration and compatibility interfaces surround an upstream-derived native core: the four C++ files retain their Jamie Zhu headers and the separate WS-DREAM/CUHK MIT notice. The author does not acquire or regrant upstream ownership through the root MIT text.

The retained WS-DREAM MIT notice, RA-QoS supplement MIT notice and DejaVu terms are byte-identical to the verified candidate. The root MIT grant does not replace them or cover datasets, separately installed dependencies or compiler runtimes.

**Conservative RA-QoS exclusion:** the recovered supplement contains an MIT grant naming Di Wu, but inspected evidence does not establish that holder's relationship to Fu, Li and Wang. This is an unresolved provenance issue, not a finding that the MIT notice is invalid or that the published method is defective. To avoid relying on that unresolved source lineage for redistribution, this publication copy omits `pcf/raqos.py`, `pcf/raqos_training.py`, `configs/raqos.json`, `requirements-raqos.txt` and `tests/test_raqos.py` in full. It does not rewrite or independently replace the affected scientific functions. The unchanged CLI's two optional RA-QoS commands are consequently unavailable. Public scalar summaries, plots and the academic citation remain unchanged; they are original reporting of the historical independent baseline. The supplement notice remains as attribution history and does not license excluded files back into the tree.

`source_manifest.json` is preserved as a historical lineage record of the original candidate. Entries for excluded RA-QoS destinations do not mean those files are included here. The historical Figure 1 entries also precede the documented DejaVu update. The actual publication file list is established by the reviewed initial commit.

## Figure fonts and third-party notices

All three public figure PDFs embed DejaVu Sans subsets. Figure 1's public renderer and editable draw.io source use DejaVu Sans, including bold and oblique mathematical glyphs; its PDF/SVG/PNG were regenerated on 2026-11-20. Figure 1 SVG text is converted to vector paths. PNGs are rendered graphics. No standalone `.ttf`, `.otf` or font package is distributed. Embedded font software is not relicensed under Paper5's MIT grant. The manuscript V3.5 and preserved earlier candidate exports were not changed.

The exact DejaVu notice supplied with the tested Matplotlib 3.8.4 installation is retained in [LICENSES/DejaVu-fonts.txt](LICENSES/DejaVu-fonts.txt), including Bitstream and Arev notices. The [upstream DejaVu license](https://dejavu-fonts.github.io/License.html) distinguishes font-software rights from the original figure content.

For the historical Arial export, the inspected OpenType `fsType=8` (`0x0008`) denotes **editable embedding**, not restricted embedding; restricted-license embedding uses value 2. See the [OpenType OS/2 specification](https://learn.microsoft.com/en-us/typography/opentype/spec/os2#fstype). This flag alone does not establish font acquisition entitlement. The public Figure 1 now uses Matplotlib's bundled DejaVu Sans (inspected regular/bold/oblique files: `fsType=0`), with the existing complete font notice retained. The current candidate therefore no longer depends on Arial availability or entitlement for rendering or its embedded fonts. This change makes no assertion about historical or manuscript exports.

## Publication review boundary

The author's MIT approval for rights they hold is recorded above; it is no longer pending. RA-QoS supplement ownership clarification remains unresolved, with the optional executable component excluded conservatively. No dataset or dependency/runtime redistribution is authorized by this source-only license.

Preparation of this copy does not authorize a public push. The author must review the documentation/licensing changes and five-file exclusion, and resolve the separately identified publication-policy question before publication. No scientific code, configuration values, figure assets, manuscript results or saved scalar values were changed.
