# MaSTNeT: Masked Spatio-Temporal Neighbourhood Transformer for Imputation of Sparse, Tabular Spatio-temporal data with uncertainty

Code accompanying the paper *Jenniges et al. 2026*
on imputation of sparse physical and biogeochemical ocean in-situ observations. We introduce **MaSTNeT** (Masked Spatio-Temporal Neighbourhood Transformer), a transformer that attends to k-nearest spatial and temporal neighbours to impute missing values. We benchmark MaSTNeT against a range of classical and deep-learning baselines on the [COMFORT dataset](https://doi.org/10.11582/2022.00039).

**Imputed variables:** Potential temperature, salinity, oxygen, nitrate, silicate, phosphate

## MaSTNeT

MaSTNeT builds pre-computed spatio-temporal k-NN graphs over all observations and reconstructs each query point by attending to its nearest neighbours in separate spatial and temporal graphs. Key design choices:

- **Separate neighbourhood graphs** For space and time (cross-attention per scope)
- **Heteroscedastic uncertainty**: The model outputs both a mean and a per-point variance, quantifying aleatoric uncertainty
- **Observation mask as input**: Missing-value indicators are appended to each token, so the model knows which features are truly observed vs. zero-filled
- **Masked training objective**: Masking was applied randomly per sample and according to geographic blocks, forcing the model to reconstruct the hidden values from context

## Installation

**Conda environment (includes experiment setup for the paper):**
```bash
conda env create -f environment.yml
conda activate time_imputation
```

**Pip (standalone MaSTNetImputer only):**

The sklearn-compatible `MaSTNetImputer` lives in the companion package [mastnet-imputer](https://github.com/y-jenniges/mastnet-imputer):
```bash
pip install mastnet-imputer
```

## Reproducing the paper experiments

### 1. Data preparation

Download the COMFORT dataset (see [Data](#data) below) and place it at `../../data/comfort.sqlite` relative to the repo root (or update `source_db_path` in `config.py`). Then generate the gridded dataset and the cross-validation splits:

```bash
python prepare_and_split/generate_splits.py
```

This writes preprocessed CSVs to `output/gridding/` and split index files to `output/splits/`.

### 2. Running the tuning study

**Ablation / single-split run**:
```bash
python experiments/mastnet_test.py --exp_id 0
```

`--exp_id` selects a named `ModelConfig` entry from `experiments/modelconfig.py` (e.g. `exp0`–`exp90+`). Results are written to `output/tuning/mastnet/architecture_ablation/`.

**Optuna hyperparameter search** (runs across all 5 selected splits):
```bash
python experiments/pytorch_study.py --model_name mastnet --n_trials 50
```

Results (JSON metrics, loss plots, saved weights) are written to `output/tuning/`.

### 3. Evaluation

Evaluation notebooks in the repository root analyse results and produce the paper figures:

| Notebook | Content                         |
|---|---------------------------------|
| `test_evaluation.ipynb` | Main results table, per-variable metrics |
| `ablation_evaluation.ipynb` | Ablation study across ModelConfig experiments |
| `tuning_evaluation.ipynb` | Hyperparameter analysis         |
| `uncertainty_analysis.ipynb` | Aleatoric and epistemic uncertainty |
| `data_exploration.ipynb` | Exploratory analysis of the COMFORT dataset |
| `woa_comparison.ipynb` | Comparison against World Ocean Atlas climatology |
| `cmip6_comparison.ipynb` | Comparison against CMIP6 model output |

## Repository structure

```
models/              Model definitions (MaSTNeT, MLP)
nn_utils/            Training infrastructure (Trainer, datasets, graph, losses)
utils/               Metrics, preprocessing, plotting utilities
experiments/         ModelConfig ablation experiments; per-model tuning scripts
prepare_and_split/   Data gridding and cross-validation split generation
config.py            Global paths, parameter names, grid configuration
output/              Generated outputs (gitignored: gridding/, splits/, tuning/)
```

The sklearn-compatible `MaSTNetImputer` is developed as a separate, installable package in the companion repository [mastnet-imputer](https://github.com/yvjennig/mastnet-imputer).

## Data

- **COMFORT dataset** (ocean in-situ observations):
  Korablev, A., Olsen, A., Geophysical Institute, University of Bergen, Bjerknes Centre for Climate Research (2022). *COMFORT Dataset*. Norstore. https://doi.org/10.11582/2022.00039

- **Bathymetry** (GEBCO 2022, used for land masking):
  GEBCO Bathymetric Compilation Group 2022 (2022). *The GEBCO_2022 Grid - a continuous terrain model of the global oceans and land.* NERC EDS British Oceanographic Data Centre NOC. https://doi.org/10.5285/e0f0bb80-ab44-2739-e053-6c86abc0289c

## Citation

If you use this code in your work, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
}
```

## References

Some utility code is adapted from:
Jenniges et al. (2025). Unveiling 3D ocean biogeochemical provinces in the North Atlantic: A systematic comparison and validation of clustering methods. *Ecological Informatics*, 91, 103390. https://doi.org/10.1016/j.ecoinf.2025.103390
