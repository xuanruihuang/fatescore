# Code release for the ICML submission: reproduction map

Paper: `icml/main.tex`, "Does Agreement with Clonal Fate Measure Inferred Cell Dynamics?".
Entry point: `reproduce.sh` in the release root.

```
bash reproduce.sh [ROOT]          # ROOT defaults to the folder holding the script
SKIP_STB_ARMS=1 bash reproduce.sh # skip a stage
RUN_PRESCIENT_REPRO=1 bash reproduce.sh  # optional stages: PRESCIENT_REPRO, COSPAR
```

Every stage writes `logs/<STAGE>.log`; results go to `out/`, figures are copied to `icml/figures/`.
The data sources, with checksums where they were recorded, are listed in the header of
`reproduce.sh`. Only the LARRY deposit and the CellTag-multi haematopoiesis clone table are
downloaded by the script (as `reproduce.sh` does); every other input is checked for presence and
the script prints where to get it.

The chained script has not been executed end to end. Each command in it is the call recorded in
`README.md`, in a server runner (`code/run_*.sh`), or the script's documented usage; where it was
not recorded, the stage is a commented TODO (list at the end). A failure is more likely a
chaining bug than an analysis bug.

## Paper element -> stage -> output

Stage names are those of `reproduce.sh`. Paths are relative to ROOT. Tables 1-5 are typeset
by hand from the listed files; no script writes the LaTeX.

### Figures

| element | stage(s) | script(s) | output |
|---|---|---|---|
| Figure 1 (`fig:concept`), page one | TABLE_GAIN, FIG2, FIG_CONCEPT | `113_fig_concept.py` | `out/fig_concept.pdf` (panel c from `out/calibrated_table.csv`, band from `out/fig_gain_band.txt`; panels a, b schematic, fixed seed) |
| Figure 2 (`fig:published`), published LARRY test set | PUBLISHED_TEST, FIG1 | `101_weinreb_test_set.py`, `101b_fig_published.py` | `out/fig_published.pdf` (from `out/weinreb_test_scores.csv`); open markers are the published PRESCIENT values, hard-coded in `101b` (`PRESCIENT_PUB`, Yeo et al. 2021) |
| Figure 3 (`fig:gain`), gain over the kernel | COSPAR, USB_PACE, PACE_PICK, SCORE, SIMULATION, TABLE_GAIN, FIG2 | `51`, `97`, `98`, `99_calibrated_table.py`, `99b_fig_gain.py` | `out/fig_gain.pdf` (from `out/calibrated_table.csv`, band from `out/mechanism_sim.csv`; band printed to `out/fig_gain_band.txt`) |
| Figure 4 (`fig:source`), where the score comes from | LARRY_EXPR_REF, LARRY_KERNELS, TABLE_GAIN, FIG_SOURCE | `03_evaluate.py`, `29_expression_kernel.py`, `115_fig_source.py` | `out/fig_source.pdf` (classifier rows from `out/ev_d2_summary.csv` and `ev_d2_bootstrap.csv`; walk from `out/feat_d2`, `out/feat_all`; methods from `out/calibrated_table.csv`) |
| Figure 5 (`fig:mechanism`), simulated positive control | SIMULATION, RESOLUTION, FIG3 | `98_mechanism_sim.py` (three grids), `102_resolution.py`, `103_fig_mechanism.py` | `out/fig_mechanism.pdf` (from `out/mechanism_sim.csv`, `mechanism_sim_noise.csv`, `mechanism_sim_wrong.csv`, `resolution.csv`) |
| Figure 6 (`fig:heldout`), held-out time point against fate | HELDOUT_LEARNED, USB_PACE, TABLE_GAIN, FIG_HELDOUT | `106_heldout_summary.py`, `114_fig_heldout_fate.py` | `out/fig_heldout_fate.pdf` (from `out/heldout/summary_110.csv` and `out/calibrated_table.csv`) |

### Tables

| element | stage(s) | output |
|---|---|---|
| Table 1 (`tab:setup`): LARRY 567 clones / 293 neutrophil-dominated | TABLE_GAIN | `out/calibrated_table.csv` (`n_clones`, `n_pos`, LARRY rows) |
| Table 1: CellTag-multi haem. 667 / 533; reprog. 304 / 112 | CTM_BUILD, CTR_BUILD, SCORE | `out/ctm_fate_scores.csv`, `out/ctr_fate_scores.csv` (`n_clones`, `n_pos_clones`); also printed in `logs/CTM_BUILD.log`, `logs/CTR_BUILD.log` |
| Table 1: methods and data sets | - | text |
| Table 2 (`tab:ladder`), clonal fate row (0.785, 0.776, 0.739, 0.810) | LADDER | `out/ladder_fate.csv`, column `auroc_clone`, rows `nm_cr_lad_velocity_dyn`, `nm_cr_lad_shuffled`, `nm_cr_lad_degradation`, `nm_cr_lad_connectivity` |
| Table 2, labelling haem. and RPE1 rows; cell-cycle row | LABELLING | `out/labelling_hsc.csv`, `out/labelling_rpe1.csv` (reference `production`, gene space; cell cycle: reference `cycle`) |
| Table 2, labelling organoid row | TODO | not reproduced by the script (see TODOs) |
| Table 3 (`tab:heldout`), USB and PACE rows | USB_PACE | `out/heldout/summary_110.csv` |
| Table 3 (`tab:heldout`), learned models and references | HELDOUT_LEARNED | `out/heldout/summary.csv` (`gap_closed_mean/min/max` x 100; reference rows `ot` = exact OT interpolation, `last` = last day alone); per-seed inputs `out/heldout/{larry,ctr}_*.csv` |
| Table 4 (`tab:fit`), flow matching | FLOW_MATCHING | `out/feat_{all,ctm,ctr}/snap_{otcfm,sbcfm}_g50k_log.json` and `logs/FLOW_MATCHING.log` (fit per seed, floor, no movement) |
| Table 4, scNODE, scIMF, PI-SDE, MIOFlow; no-mass counts in parentheses | STB_ARMS | `out/feat_{all,ctm,ctr}/stb_<method>_log.json` (`fit`, `fit_reference`); no-mass counts in `logs/STB_ARMS.log` |
| Table 4, USB and PACE | USB_PACE | `out/feat_{all,ctm,ctr}/snap_usb_run_log.json`, `snap_pace_run_log.json`, `snap_pace_ipsc_run_log.json` (`fit`, `fit_reference`); PACE row is the preset kept by PACE_PICK |
| Table 4 note: scIMF at 1,000 epochs (16.4) | TODO | variant arguments not recorded |
| Table 5 (`tab:labelling`), haem. and RPE1 rows | LABELLING | `out/labelling_hsc.csv`, `out/labelling_rpe1.csv` (best velocity, best field without dynamics, paired difference with bootstrap interval) |
| Table 5, organoid rows | TODO | not reproduced by the script |

### In-text results

| where | result | stage | output |
|---|---|---|---|
| Abstract, Sec. 3.2, Contributions | 48 pairs, 18 methods: 36 below, 12 not separated, 0 above | TABLE_GAIN | `out/calibrated_table.csv` (`separated`) |
| Abstract, Sec. 3.1 | kernel r 0.33-0.42 on the 335 test cells; FateID/PBA/WOT 0.15-0.23 | PUBLISHED_TEST | `out/weinreb_test_scores.csv` |
| Abstract, Sec. 3.1 | PRESCIENT published 0.41 (0.407 +- 0.019), 0.399 +- 0.025 with weights, 0.196 +- 0.020 without | - | quoted from Yeo et al. 2021, not computed |
| Abstract, Contributions, Sec. 5 | true dynamics add 0.05-0.10 (0.048-0.097) at priming 0.2 | SIMULATION, FIG2 | `out/mechanism_sim.csv`; `out/fig_gain_band.txt` |
| Abstract, Sec. 6 | held-out time point separates four learned models (scNODE, OT-CFM, SB-CFM, USB) | HELDOUT_LEARNED, USB_PACE | `out/heldout/summary.csv`, `out/heldout/summary_110.csv` |
| Contributions, Sec. 4 | day-2 expression classifier AUROC 0.870; sorted population alone 0.625 | LARRY_EXPR_REF | `out/ev_d2_summary.csv` (total-count and population-only rows; carried over from the preprint analysis) |
| Sec. 2 | kernel at 18 graph settings, reference n30 p30 with density normalisation | LARRY_KERNELS, CT_KERNELS | `out/feat_all/cr_conn_n*_p*_d*.npy`; `out/feat_{ctm,ctr}/cr_conn_all_n*_p*_d*.npy` |
| Sec. 2, App. C | 2,000 paired bootstrap resamples, MDD = 0.714 x interval width | - | `97_fate_score.py`, `99_calibrated_table.py` (code) |
| Sec. 3.1 | FateID 0.225, WOT 0.150, upper bound 0.485; 335 cells / 298 clones matched by barcode; shares equal the published targets | PUBLISHED_TEST | `logs/PUBLISHED_TEST.log`, `out/weinreb_test_scores.csv` |
| Sec. 3.1 | kernel 0.371 at the reference; all 18 methods below or not separated; StationaryOT, CoSpar and velocity kernels 0.26-0.33 | PUBLISHED_TEST | `out/weinreb_test_scores.csv` |
| Sec. 3.1, App. A | faithful PRESCIENT runs 0.16 (full data, KEGG, five seeds) | PRESCIENT_REPRO (optional) | `out/prescient_check/full_kegg_s*/result_full_kegg_seed*.json` |
| Sec. 3.2 | nearest to zero: real-time kernel -0.001, StationaryOT -0.011 (LARRY); scIMF -0.003, CoSpar +0.003, StationaryOT +0.008 (reprog.); farthest scIMF LARRY -0.205 (USB LARRY -0.206 in Sec. 5) | TABLE_GAIN | `out/calibrated_table.csv` (`gain`, `lo`, `hi`) |
| Sec. 3.2 | kernel settings move it 0.010-0.021; MDD 0.013-0.077 | RESOLUTION | `out/resolution.csv` (also `kernel_min/max` in `calibrated_table.csv`) |
| Sec. 3.2, Sec. 6 | scNODE fit 3.9 vs floor ~3.6, gain -0.147; PI-SDE fit 5.0, gain -0.068 | STB_ARMS, TABLE_GAIN | `out/feat_all/stb_scNODE_log.json`, `stb_PISDE_log.json`; `out/calibrated_table.csv` |
| Sec. 3.2 | second kernel setting adds at most 0.003; scNODE haem. +0.031 [0.008, 0.057] | INCREMENT | `out/incremental.csv` |
| Sec. 3.2 | 157 of 889 task cells without terminal mass; +0.048 [0.019, 0.078] without them | INCREMENT | `logs/INCREMENT.log` (printed by `104b_nomass_check.py`); uncorrected p ~ 0.01 read from the interval |
| Sec. 4 | kernel AUROC 0.81 on LARRY | TABLE_GAIN | `out/calibrated_table.csv` (`kernel_auroc` 0.808) |
| Sec. 4 | haem. early-cell graph 0.830 vs all cells 0.824 | CT_KERNELS, SCORE | `out/ctm_fate_scores.csv` rows `cr_conn_early_n30_p30_d1`, `cr_conn_all_n30_p30_d1` |
| Sec. 4 | LARRY day-2-only graph lowers the kernel to 0.76-0.78 | LARRY_KERNELS | `out/refint_d2*` (`67_reference_intervals.py` over `out/feat_d2/cr_conn_*` from `29 --d2-only`) |
| Sec. 4 | ladder 0.785 / 0.776 / 0.739 / 0.810 | LADDER | `out/ladder_fate.csv` |
| Sec. 5 | kernel AUROC 0.49, 0.67, 0.83, 0.98 by priming; gain at most 0.02 at 0.98 | SIMULATION | `out/mechanism_sim.csv` |
| Sec. 5 | all 48 pairs MDD < 0.097, 18 < 0.048 | RESOLUTION | `out/resolution.csv`, `logs/RESOLUTION.log` |
| Sec. 5 | random error erodes the gain towards zero; shuffled velocity at or above the kernel | SIMULATION | `out/mechanism_sim_noise.csv` |
| Sec. 5 | uniform decay only shrinks the gain; delta = -0.5 gives -0.02 to -0.05 | SIMULATION | `out/mechanism_sim_wrong.csv` |
| Sec. 5 | 36 pairs below, by up to 0.21 | TABLE_GAIN | `out/calibrated_table.csv` |
| Sec. 6 | labelling: six of six comparisons separate | LABELLING (+ organoid TODO) | `out/labelling_{hsc,rpe1}.csv` |
| Sec. 6 | cell cycle 0.439 vs neighbour average 0.263 | LABELLING | `out/labelling_rpe1.csv` (reference `cycle`) |
| Sec. 6 | scNODE, OT-CFM, SB-CFM, USB close 29-35% on LARRY; exact OT 29%; PI-SDE 14% on reprog.; PACE below independent pairing | HELDOUT_LEARNED, USB_PACE | `out/heldout/summary.csv`, `out/heldout/summary_110.csv` |
| Sec. 6 | Waddington-OT's interpolation closes 59-81% (LARRY), 9-12% (reprog.) | HELDOUT | `out/ot_interpolation.csv`, `out/ot_interpolation_ctr.csv`; share = (R - I) / (R - P) from Waddington-OT's validation summary (names R, I, P), per seed; no script in `code/` writes the range (`93_ladder.py` applies the formula to LARRY only and reads preprint outputs) |
| App. A | StationaryOT reprog. tolerance reached 3.0e-5 vs 5.6e-5 | OT_ARMS | `out/feat_ctr/snap_statot_log.json` |
| App. A, App. D | full labelling reference has cosine 0.48 with its own decay term; organoid gene space 0.092 vs 0.114 | TODO (organoid) | not reproduced by the script |
| App. A (i) | our first PRESCIENT run with the published read-out: r 0.06 | PRESCIENT_REPRO | `out/prescient_check/resim_none.npz`, `logs/PRESCIENT_REPRO.log` |
| App. A (ii) | published LARRY settings, no weights: 0.054 +- 0.010 | PRESCIENT_REPRO | `out/prescient_check/repro_none_s*/result_none_seed*.json` |
| App. A (iii) | full data 0.082 +- 0.019 / 0.159 +- 0.017; KEGG on our object 0.174 +- 0.016; main-task gains -0.06 to -0.16; 40% reach neither fate | PRESCIENT_REPRO | `out/prescient_check/full_{none,kegg}_s*/result_full_*_seed*.json`, `repro_kegg_s*/result_kegg_seed*.json`, `logs/PRESCIENT_REPRO.log`; mean +- SD over seeds computed from the per-seed files (no aggregation script) |
| App. A, version items | torch 2.8 / CUDA 12.8 environment; MIOFlow 1c6ceb4; PI-SDE on GPU | STB_INSTALL | `code/install_sctimebench.sh`, `code/install_mioflow_old.sh`; `ext/env_stb_freeze.txt` |
| Sec. 3.2, App. C | recalibrated Brier: no method below the kernel in 48 pairs (24 above, 24 not separated) | BRIER | `out/brier.csv` (`better_cal`; rows `kernel, other setting` are sanity rows) |
| App. C | six terminal-state definitions: walk 0.78-0.82, velocity kernel below under every definition | TERMINAL_SIM | `out/terminal_states_alt.csv` |
| App. C | trifurcation: true dynamics add +0.06 to +0.12; growth changes gains by <= 0.001; shuffled +0.01 to +0.02 | TERMINAL_SIM | `out/mechanism_multi.csv`, `out/mechanism_multi_noise.csv` |
| App. C | clones whose present state leaves fate open (thirds of the classifier margin) | HARD_CELLS | `out/hard_cells_{larry,ctm,ctr}.csv` |
| App. C | the walk plus a little direction: split-validated gains, real-time kernel +0.018 [0.002, 0.033] | MIX | `out/mix111_scores.csv`, `out/mix111_ctm_scores.csv` |
| Sec. 3.2, App. C | read-out: none of 27 learned-model pairs above the walk on fully reached clones | READOUT | `out/readout_check.csv` |
| Sec. 6, App. E | full labelling reference: five of six comparisons separate; organoid gene space does not | LABELLING (+ organoid TODO) | `out/labelling_{hsc,rpe1}.csv`, rows with reference `labelling`; organoid from `out/labelling_organoid.csv` (not yet produced by the script) |
| Abstract, Contributions, Sec. 3.2, App. C | six fates on LARRY (771 clones): walk 0.893 (0.879-0.903); learned models -0.13 to -0.22; StationaryOT +0.012 [0.003, 0.024], p 0.001, Holm 0.009, above 12 of 18 settings; megakaryocytes +0.060 [0.010, 0.126]; four fates with >= 30 clones +0.003 | MULTIFATE | `out/multifate117_larry.csv`, `out/multifate117_larry_perfate.csv`, `logs/MULTIFATE.log` |
| App. C | scoring tool (kernel, band, gain, MDD, increment) | TOOL_CHECK | `tool/tests/test_core.py`; reproduces the LARRY StationaryOT gain -0.011 (-0.023 to 0.002) and increment 0.001 |

CoSpar (stage COSPAR), USB and PACE (stage USB_PACE) are scored in the paper and run before PACE_PICK and SCORE, in the
order of `code/run_fold3.sh`.

Test run (2026-11-26): the light stages (PACE_PICK, SCORE, PUBLISHED_TEST, FIG1, TABLE_GAIN, FIG2,
RESOLUTION, FIG3, INCREMENT, BRIER, FIG_CONCEPT, TOOL_CHECK, FIGURES_COPY) ran end to end in an isolated
root on copies of the method outputs; `calibrated_table.csv`, `weinreb_test_scores.csv`,
`resolution.csv`, `incremental.csv`, `brier.csv` and both CellTag-multi score tables were identical to the
live ones (largest difference 1.6e-8). Method-training stages were not rerun.

## Software versions

| environment | used by | versions (as recorded) |
|---|---|---|
| `env` (analysis) | 02-03, 29, 51, 66, 67, 70, 92, 97-104, 105_diag, 106_summary, 107_score, tool | python 3.11.16, numpy 2.4.6, scipy 1.16.3, pandas 2.3.3, scikit-learn 1.9.1, anndata 0.12.19, scanpy 1.11.5, scvelo 0.3.4, velovi 0.3.1, scvi-tools 1.4.2, cellrank 2.0.7, matplotlib 3.11.2, torch 2.11.0 (+cu130), loompy, igraph, pygam (unpinned). Build commands in `README.md` ("Reproducing") |
| `env_ot` | 41, 43, 90, 94, 105 (except diag), 106_refs/cfm | python 3.11.16; StationaryOT (statot) 0.0.14 (imported through a CellRank-1 shim in `41`); wot 1.0.8.post2; moscot 0.5.2 with jax 0.10.2 (CPU); PRESCIENT (installed, version not recorded; `105` follows repository gifford-lab/prescient commit 50971c7); numpy 2.4.6, scipy 1.17.1, POT 0.9.7.post1, anndata 0.12.19, scanpy 1.11.5; torchcfm 1.0.7 (94). No build script in `code/` |
| `ext/env_stb` | 100, 106_stb | venv on env_ot's python 3.11; scNODE's pinned requirements (torch 2.8, CUDA 12.8), POT, torchsde, trampoline, boltons, Squidiff, scTimeBench (editable; commit not recorded); MIOFlow replaced by commit 1c6ceb49841b17d178d4795439182b57250b467d (phate, graphtools, torchdiffeq, seaborn added). `install_sctimebench.sh` writes the full `pip freeze` to `ext/env_stb_freeze.txt` |
| `env_dyn` | 91, 91b | dynamo 1.5.3, scvelo 0.3.4. No build script in `code/` |
| `ext/env_cospar` | 107_state | cospar 0.6.0 with numpy 1.26.4, scipy 1.13.1, numba 0.60, scikit-learn 1.5.2, umap 0.5.7, scanpy 1.10.4 (freeze in `ext/env_cospar_freeze.txt`) |
| `ext/env_usb` | 110_usb | USB clone at commit 0cad29b (github.com/Sirin6/USB; no licence: used for research, not redistributed); torch with RTX 5090 kernels; set up by hand |
| `ext/env_pace` | 110_pace | PACE clone at commit 5f5a10e (github.com/AI4Science-WestlakeU/PACE, MIT); its requirements plus attrs and torchcde; set up by hand |

## Hardware and wall time (where recorded)

- GPU: one CUDA device for veloVI (`02_features.py`); NVIDIA RTX 5090 for the scTimeBench
  methods (hence the single torch 2.8 / CUDA 12.8 environment); GPU also for PRESCIENT, flow
  matching and the 106 held-out runs.
- LARRY feature stage about 50 min (the dynamical fit on 49,302 cells is 13 min on 16 cores);
  evaluation stage (`03_evaluate.py` and the preprint evaluations) about 2 h on 28 cores; peak
  memory about 75 GB (`reproduce.sh`, `README.md`).
- StationaryOT on reprogramming: the first run at tolerance 1e-9 ran 3.5 h and ended at 1.08 times
  the tolerance without saving; rerun at 2e-9 (`run_ctr_statot2.sh`).
- Disk: about 3 GB for the LARRY deposit, 4.0 GB for `GSE216521_RAW.tar`, 2.07 GB for the
  GSM4185642 normalised counts (optional stage).
- No wall time was recorded for the other stages.

## TODO stages and reconstructed commands

Written as commented TODOs in `reproduce.sh` (arguments not recoverable from docstrings or
`README.md`):
- `91_labelling_velocity.py` on the scEU-seq intestinal organoid (Table 2 row, Table 5 rows 5-6,
  the 0.48 / 0.092 vs 0.114 statements): input file and arguments not recorded.
- `100_sctimebench_fate.py` scIMF 1,000-epoch variant (Table 4 note): `--set`/`--tag` not recorded.
- Input preparation not scripted in `code/`: `data/labelling/{hsc_small,hsc_published_velocity,
  rpe1_small}.h5ad` (from dynamo's sample data); `data/kegg/*.csv` (KEGG REST mmu04110, mmu04210).

Commands written from the scripts' documented usage and defaults, because the exact call was not
recorded: `41_ot_fate.py --method statot|wot|moscot` (script defaults) on all three data sets;
`43_prescient_fate.py` on all three; `94_flow_matching_fate.py --tag _g50k --scale global --steps
50000` on the CellTag-multi data (the tag is the one `99_calibrated_table.py` reads); `96_conn_kernel.py`
modes; `29_expression_kernel.py` on `feat_all`; `90_ot_interpolation.py` on reprogramming; the
noise and wrong-direction grids of `98_mechanism_sim.py` (taken from the reported tables); the output
directories of 43, 105 and 107. Environment builds for `env_ot` and `env_dyn` are not scripted.

