<h1 align="center">
<p> MDCP
</h1>

This repo serves as the implementation for the paper ["Multi-Distribution Robust Conformal Prediction"](https://arxiv.org/abs/2601.02998).

![Visual Overview](assets/visual.png)

> **Table of Contents**
> - [Installation](#installation)
> - [Simulation experiments](#simulation-experiments)
> - [Real-data applications](#real-data-applications)
> - [Figures](#figures)
> - [Repo layout](#repo-layout)


## Installation

**1. Create & activate an environment**

Windows (PowerShell)
```powershell
python -m venv mdcp
.\mdcp\Scripts\Activate.ps1
```

macOS/Linux (bash/zsh)
```bash
python -m venv mdcp
source mdcp/bin/activate
```

**2. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt wilds
```


## Simulation experiments

All simulation scripts live under `notebook/` and write artifacts to `eval_out/` (see `model/const.py` for exact folders). The paper reports averages over 100 independent runs per configuration over contiguous seed ranges starting from the base seed.

Use the following commands to run the simulations under different suites. Each writes its outputs to `eval_out/<suite_name>/`.

```bash
# Linear suite
python notebook/eval_linear.py --num-trials 1 --base-seed 34567

# Nonlinear suite
python notebook/eval_nonlinear.py --num-trials 1 --base-seed 23456

# Temperature suite
python notebook/eval_temperature.py --num-trials 1 --base-seed 12345 --temperatures 0.5 1.5 2.5 3.5 4.5

# Covariate shift suite
python notebook/eval_cov_shift.py --num-trials 1 --base-seed 34567 --delta-xs 0 0.5 1.5 2.5 3.5 4.5

# Covariate & concept shift suite
python notebook/eval_cov_and_concept_shift.py --num-trials 1 --base-seed 45678 --delta-xs 0 0.5 1.5 2.5 3.5 4.5
```



## Real-data applications

Three real-world data evaluations are supported: [FMoW](https://wilds.stanford.edu/datasets/#fmow), [PovertyMap](https://wilds.stanford.edu/datasets/#povertymap) from [WILDS](https://github.com/p-lambda/wilds), and [MEPS](https://meps.ahrq.gov/mepsweb/).

The evaluation scripts are under `meps`, `wilds/fmow`, and `wilds/poverty`. Reported paper results average 100 runs per dataset using contiguous seed ranges starting from the base seed.

**Prerequisites:**

1.  Clone the upstream WILDS repo (provides the `examples/` modules used by Poverty/FMoW):
    ```bash
    python -m wilds.setup_wilds_repo
    ```
    The helper places the checkout at `external/wilds_upstream`; pass `--wilds-repo` if you keep it elsewhere.

2.  Download WILDS datasets (stores under `data/`):
    ```bash
    python -m wilds.download_datasets --dataset poverty --root data --unpack
    python -m wilds.download_datasets --dataset fmow --root data --unpack
    ```
    If TLS inspection blocks the download, run `python wilds/download_wilds_insecure.py --dataset <name> --root data --unpack` and move the extracted folders under `data/`.

3.  Prepare CSVs for MEPS

    Follow [the download guide](https://github.com/yromano/cqr/tree/master/get_meps_data), then drop the three cleaned files into `meps/data/` with names `meps_*_reg.csv`.

**MEPS**

```bash
# Run the evaluator, outputs default to `eval_out/meps/`
python -m meps.eval --panels 19 20 21 --num-trials 1 --base-seed 45678
```

**PovertyMap**

```bash
# 1. Create 2014-2016 split
python wilds/poverty/create_training_split.py --repo-root . --train-frac 0.375 --seed 0 --years 2014 2015 2016

# 2. Train ResNet18 multi-head regressor
python wilds/poverty/train_resnet.py --repo-root .

# 3. Predict on the holdout set
python wilds/poverty/predict_density.py --repo-root . --checkpoint eval_out/poverty/training/run_multihead_gaussian/best_model.pth

# 4. Learn lambda, calibrate and evaluate
python wilds/poverty/analysis/eval.py --prediction-dir eval_out/poverty/predictions/run_multihead_gaussian --num-trials 1 --base-seed 90000

# 5. Plot results, including preprocessing required to reproduce the paper’s figures
python wilds/poverty/analysis/plot.py
```

**FMoW**

```bash
# 1. Create 2016-only splits
python wilds/fmow/pipeline/cli.py split --root data/fmow_v1.1 --wilds-repo external/wilds_upstream --output eval_out/fmow/splits/2016_region --train-frac 0.375 --seed 0

# 2. Train DenseNet121 on the training set
python wilds/fmow/pipeline/cli.py train --root data/fmow_v1.1 --wilds-repo external/wilds_upstream --train-idx eval_out/fmow/splits/2016_region/train_idx.npy --holdout-idx eval_out/fmow/splits/2016_region/holdout_idx.npy --output eval_out/fmow/training/run_densenet121 --epochs 30 --batch-size 32

# 3. Predict on the holdout set
python wilds/fmow/pipeline/cli.py predict --root data/fmow_v1.1 --wilds-repo external/wilds_upstream --holdout-idx eval_out/fmow/splits/2016_region/holdout_idx.npy --checkpoint eval_out/fmow/training/run_densenet121/checkpoint_best.pt --output eval_out/fmow/predictions/run_densenet121 --batch-size 128 --save-embeddings

# 4. Learn lambda, calibrate and evaluate
python wilds/fmow/analysis/eval.py --prediction-dir eval_out/fmow/predictions/run_densenet121 --num-trials 1 --base-seed 70000

# 5. Plot results, including preprocessing required to reproduce the paper’s figures
python wilds/fmow/analysis/plot.py
```

> Use `--wilds-repo` if your upstream checkout is not at `external/wilds_upstream`.



## Figures

Run these commands from the repo root after the corresponding evaluation artifacts exist. Outputs will be saved under `eval_out/paper_figures/`.

Reproduce simulation plots:

```bash
# Linear
python notebook/paper_figures/plot_linear.py --eval-dir eval_out/linear --output-dir eval_out/paper_figures

# Nonlinear
python notebook/paper_figures/plot_nonlinear.py --eval-dir eval_out/nonlinear --output-dir eval_out/paper_figures

# Temperature
python notebook/paper_figures/plot_temperature.py --eval-root eval_out/temperature --output-dir eval_out/paper_figures

# Covariate shift
python notebook/paper_figures/plot_cov_shift.py --eval-root eval_out/cov_shift --output-dir eval_out/paper_figures

# Covariate + concept shift
python notebook/paper_figures/plot_cov_and_concept_shift.py --eval-root eval_out/cov_and_concept_shift --output-dir eval_out/paper_figures
```

Reproduce real-data plots:

```bash
# PovertyMap
python notebook/paper_figures/plot_poverty.py --input-dir eval_out/poverty/mdcp --output-dir eval_out/paper_figures

# PovertyMap target distribution
python notebook/paper_figures/plot_poverty_target.py --repo-root . --output eval_out/paper_figures/poverty_target.pdf

# FMoW
python notebook/paper_figures/plot_fmow.py --input-dir eval_out/fmow/mdcp --output-dir eval_out/paper_figures

# MEPS
python notebook/paper_figures/plot_meps.py --input eval_out/meps --output eval_out/paper_figures --coverage-target 0.9
```


## Repo layout

```
MDCP/
├── model/
│   ├── MDCP.py
│   └── const.py
├── notebook/
│   ├── eval_linear.py
│   ├── eval_nonlinear.py
│   ├── eval_temperature.py
│   ├── eval_cov_shift.py
│   ├── eval_cov_and_concept_shift.py
│   └── paper_figures/
├── meps/
│   └── eval.py
├── wilds/
│   ├── poverty/
│   └── fmow/
├── data/ (downloaded datasets)
└── eval_out/ (experiment outputs + figures)
```


## Citation 

Please use the following bibliography for citing our method and package. 


```
@misc{yang2026multidistributionrobustconformalprediction,
      title={Multi-Distribution Robust Conformal Prediction}, 
      author={Yuqi Yang and Ying Jin},
      year={2026},
      eprint={2601.02998},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02998}, 
}
```

