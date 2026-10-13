# Figures of the paper

One script per figure. Each script reads the precomputed results in `../python/data`
and writes `<name>.pdf` and `<name>.png` to `../images`, or to the folder given in the
environment variable `FIG_OUT`.

```bash
python fig07_maps.py                      # one figure
python make_all.py                        # all figures
python make_all.py fig03 fig11            # some figures
FIG_OUT=/tmp/test python make_all.py      # write somewhere else (to compare before overwriting)
```

| script | figure | output | data (`../python/data`) | produced by (`../python`) |
|---|---|---|---|---|
| `fig01_schematic.tex` | 1 `fig:schematic` | `fig1_schematic.pdf` | – (TikZ) | `pdflatex -jobname fig1_schematic fig01_schematic.tex` |
| `fig02_model.py` | 2 `fig:model` | `fig2_model` | `maps_20x20.pkl` | `sweep_maps.py 20 20` |
| `fig03_verification.py` | 3 `fig:verification` | `fig3_verification` | `case_outer_wall.pkl` | `cases.py` |
| `fig04_histories.py` | 4 `fig:histories` | `fig4_histories` | `case_{inner,outer}_wall.pkl` | `cases.py` |
| `fig05_fields.py` | 5 `fig:fields` | `fig5_fields_outer` | `case_outer_wall.pkl` | `cases.py` |
| `fig06_collapse.py` | 6 `fig:collapse` | `fig6_collapse` | `maps_20x20.pkl`, `case_*.pkl` | `sweep_maps.py`, `cases.py` |
| `fig07_maps.py` | 7 `fig:maps` | `fig7_maps` | `maps_20x20.pkl` | `sweep_maps.py` |
| `fig08_random.py` | 8 `fig:random` | `fig8_random` | `case_*.pkl` | `cases.py` |
| `fig09_misestimation.py` | 9 `fig:mis` | `fig9_misestimation` | `mis_*.pkl` | `misestimation.py` |
| `fig10_bounded.py` | 10 `fig:bounded` | `fig10_bounded` | `case_*.pkl` | `cases.py` |
| `fig11_obs_clean.py` | 11 `fig:obsclean` | `fig11_obs_clean` | `obs3_*_T2.pkl` | `sweep_obs.py` |
| `fig12_obs_info.py` | 12 `fig:obsinfo` | `fig12_obs_info` | – (computed, ~10 s) | `lq.py`, `obs.py` |
| `fig13_obs_noise.py` | 13 `fig:obsnoise` | `fig13_obs_noise` | `obs3_*_T2.pkl` | `sweep_obs.py` |
| `fig14_intermittent.py` | 14 `fig:intermittent` | `fig14_intermittent` | `int_*_T2.pkl` | `sweep_intermittent.py` |
| `fig15_drag.py` | 15 `fig:drag` | `fig15_drag` | `drag_*_T2.pkl` | `sweep_drag.py` |
| `fig16_drag_controls.py` | 16 `fig:dragcontrols` | `fig16_drag_controls` | `drag_*_wall_T2.pkl` | `sweep_drag.py` |
| `fig17_pdrag.py` | 17 `fig:pdrag` | `fig17_pdrag` | `pdrag_*.pkl`, `pdragps_*.pkl` | `sweep_pdrag.py`, `pdrag_extra.py` |
| `fig18_ql.py` | 18 `fig:ql` | `fig18_ql` | `ql_*.pkl`, `qltuned_*.pkl`, `qldiag_*.pkl` | `ql_study.py`, `ql_diag.py` |

`figstyle.py` holds what is common to all figures: fonts and sizes (JFM style), the colour
roles (`C_UNC` uncontrolled, `C_LQR`, `C_OPT` optimum, `C_3`), the text width `W1`, the
two representative cases, the panel-label and save helpers, and the readers of the
wavenumber maps and of the output-feedback results. Edit it to change all figures at once.

Parameters that you may want to change (lifetime `TF`, noise levels, cases shown, contour
levels, ...) are constants at the top of each script.

Requirements: numpy, scipy, matplotlib (tested with Python 3.12, matplotlib 3.10). The
scripts import the computational modules of `../python` (figures 2, 12, 15 and 18 compute
some quantities on the fly).
