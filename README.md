# How optimal is linear control?

Code and data for the paper

> A. García Gutiérrez, *How optimal is linear control? Linear feedback versus unrestricted
> optimal control of transient bursts in turbulent channels* (submitted).

The linear-quadratic regulator (LQR) is the optimal controller of linear dynamics over an
infinite horizon, but the coherent structures of wall-bounded turbulence grow through a
nearly linear transient and are destroyed after a finite lifetime. The paper quantifies the
loss of linear feedback for such bursts using the eddy-viscosity Orr–Sommerfeld–Squire model
of a turbulent channel at Re_τ = 2000, and compares the LQR with the optimum over the
lifetime, computed without assuming a control law. It also considers noisy wall sensors,
intermittent bursts, a drag objective with an amplitude-proportional actuation cost, and a
quasi-linear model with mean-flow distortion.

## Contents

| folder | content |
|---|---|
| [`python/`](python) | The model, the controllers and all the studies of the paper (see [`python/README.md`](python/README.md) for a description of every file and the commands that produce each data set). |
| [`figures/`](figures) | One script per figure of the paper (see [`figures/README.md`](figures/README.md)). |

## Installation

Python ≥ 3.10 with numpy, scipy and matplotlib:

```bash
git clone https://github.com/<user>/<repository>.git
cd <repository>
pip install -r requirements.txt
```

[JAX](https://github.com/google/jax) is needed only for the quasi-linear model and the
direct policy search (`qlmodel.py`, `ql_*.py`, `pdrag_extra.py`): `pip install jax`.

## Reproducing the figures

```bash
cd figures
python make_all.py              # all figures -> ../images/*.pdf, *.png  (about 1.5 min)
python fig07_maps.py            # a single figure
```

Figure 1 is a TikZ sketch (`figures/fig01_schematic.tex`); `make_all.py` compiles it if
`pdflatex` is available.

## Reproducing the data

The commands that produce every file of `python/data` are listed in
[`python/README.md`](python/README.md#reproduce), with approximate run times. Set
`OPENBLAS_NUM_THREADS=1` before running them: with multithreaded BLAS the small dense
linear algebra of the model is about 100 times slower. Start with

```bash
cd python
python validate_port.py         # check the Python port against the MATLAB result
```

## Citation

If you use this code, please cite the paper (reference to be completed upon publication).
See also [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).
