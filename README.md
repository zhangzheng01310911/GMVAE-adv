

# EM-GMVAE for Spatiotemporal Traffic Forecasting

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22195838.svg)](https://doi.org/10.5281/zenodo.22195838)

This repository implements the revised EM-initialized graph variational
autoencoder with an analytic auxiliary-joint Rényi regularizer for
spatiotemporal traffic forecasting. It includes model training, reproducible
data splitting and preprocessing, a synthetic integration test, scripts for
retrieving historical Hong Kong traffic records, and an automatic downloader
for the processed dataset archived on Zenodo.

## Repository structure

```text
.
├── configs/
│   └── default.yaml
├── scripts/
│   ├── download_hk_history.py
│   ├── download_zenodo_dataset.py
│   ├── make_synthetic_data.py
│   ├── prepare_hk_xml.py
│   └── train.py
├── src/
│   └── gmvae/
├── tests/
│   └── test_model.py
├── EXPERIMENT_CHECKLIST.md
├── HK_DATA_DOWNLOAD_README.md
├── RUN_SERVER.md
├── pyproject.toml
└── README.md
```

Generated datasets and experiment outputs are stored under `data/`, `work/`,
`runs/`, and `logs/`; these directories do not need to be committed to Git.

## Model overview

For each detector and input window, the GRU--graph encoder produces a
node-specific diagonal-Gaussian approximate posterior

```text
q(z_i | X, A) = Normal(mu_i^q, diag((sigma_i^q)^2)).
```

During training, latent variables are sampled using the reparameterization
trick:

```text
z_i = mu_i^q + sigma_i^q * epsilon_i,
epsilon_i ~ Normal(0, I).
```

The decoder predicts a residual relative to the persistence forecast and
reconstructs the temporal hidden representation. The implemented
reconstruction loss is the positive mean-squared error

```text
L_rec = mean((H_hat - H)^2).
```

The global mixture prior is

```text
p(z_i) = sum_k pi_k^p Normal(mu_k^p, diag((sigma_k^p)^2)).
```

Training proceeds in three stages:

1. **Warm-up:** train the encoder and decoder without GMM regularization.
2. **EM estimation:** collect posterior means from the training split and run
   EM once to estimate the global GMM prior.
3. **Joint fine-tuning:** keep the estimated GMM prior fixed and optimize the
   encoder and decoder using the complete objective.

Validation and test observations are never used to estimate the GMM prior.

The analytic auxiliary-joint Rényi regularizer is

```text
R_i = logsumexp_k[
    alpha log r_ik
    + (1-alpha) log pi_k^p
    + (alpha-1) D_alpha(q_i || p_k)
] / (alpha-1).
```

Here, `r_ik` denotes the analytic responsibility associated with posterior
`q_i` and prior component `p_k`. The component-wise Gaussian Rényi divergence
is available in closed form. No Monte Carlo density-ratio estimator is used
for this regularizer. By the data-processing inequality, the auxiliary-joint
quantity upper-bounds the Rényi divergence to the marginal GMM prior.

The overall training objective is

```text
L = L_task + lambda_rec L_rec + beta(epoch) R_alpha.
```

For the forecasting experiments in this repository, `L_task` is the MAE loss.
The stochastic decoded terms use one reparameterized latent sample per forward
pass, whereas the Rényi term is analytic.

## Requirements

- Python 3.10 or later
- NumPy 1.24 or later
- PyTorch 2.1 or later
- scikit-learn 1.3 or later
- Matplotlib 3.7 or later
- `requests` for downloading raw historical XML snapshots
- pytest 7.4 or later for tests

For GPU training, install the PyTorch build appropriate for the CUDA version
reported by `nvidia-smi` using the official PyTorch installation selector.

## Installation

```bash
git clone https://github.com/zhangzheng01310911/HK-Traffic-Dataset.git
cd HK-Traffic-Dataset

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
python -m pip install requests
```

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m pytest -q
```

## Dataset

The processed Hong Kong traffic dataset used by the experiments is publicly
archived on Zenodo:

- **Record:** <https://zenodo.org/records/22195838>
- **Permanent DOI:** <https://doi.org/10.5281/zenodo.22195838>
- **Data file:** `hk_traffic_20250601_20250621.npz`
- **File size:** 31,619,290 bytes (approximately 31.6 MB)
- **MD5:** `b787ade69207abe6a7fb1c90735e66a1`

The dataset covers June 1--21, 2025 and contains 6,048 five-minute time steps,
774 traffic detectors, and three channels:

```text
0: speed_kmh
1: volume_vehicles
2: occupancy_percent
```

The data originate from the Transport Department of the Government of the
Hong Kong Special Administrative Region through DATA.GOV.HK. Full provenance,
licensing, preprocessing, quality-control information, and checksums are
included in the Zenodo record.

### Download automatically from Zenodo

From the repository root, run:

```bash
python scripts/download_zenodo_dataset.py
```

The script queries the public Zenodo API, downloads the published NPZ, verifies
its MD5 checksum, and saves it as

```text
data/hong_kong/hk_traffic_20250601_20250621.npz
```

To specify another destination:

```bash
python scripts/download_zenodo_dataset.py \
  --output /path/to/hk_traffic_20250601_20250621.npz
```

The automatic downloader uses only the Python standard library and does not
require a Zenodo access token because the record is public.

### Direct command-line download

As an alternative, use `wget`:

```bash
mkdir -p data/hong_kong

wget -c \
  "https://zenodo.org/api/records/22195838/files/hk_traffic_20250601_20250621.npz/content" \
  -O data/hong_kong/hk_traffic_20250601_20250621.npz
```

Verify the downloaded file on Linux:

```bash
md5sum data/hong_kong/hk_traffic_20250601_20250621.npz
```

The expected checksum is

```text
b787ade69207abe6a7fb1c90735e66a1
```

## Data format

The processed NPZ contains:

| Key | Type and shape | Description |
| --- | --- | --- |
| `values` | `float32 [time, nodes, channels]` | Speed, volume, and occupancy observations |
| `adjacency` | `float32 [nodes, nodes]` | Symmetric detector adjacency matrix |
| `timestamps` | `[time]` | Five-minute timestamps |
| `detector_ids` | `[nodes]` | Detector identifiers in array order |
| `channel_names` | `[3]` | Names and order of the three channels |

The train/validation/test partition is chronological 70%/20%/10%. Missing
values are imputed using statistics calculated exclusively from the training
split. Channel-wise normalization statistics are also fitted on the training
split and then applied unchanged to validation and test observations.

### Graph construction

The graph is a data-driven speed-correlation graph. Pearson correlations are
calculated between detector speed series using only the first 70% of the
chronological observations. Each detector retains up to three other detectors
with the strongest positive correlations, and the candidate graph is
symmetrized by retaining an undirected edge whenever either detector selects
the other. The graph therefore captures similarity in temporal speed dynamics
and should not be interpreted as physical road connectivity.

## Train on the published Hong Kong dataset

The following command trains one speed-forecasting run using seed 0:

```bash
mkdir -p runs/hk logs

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py \
  --data data/hong_kong/hk_traffic_20250601_20250621.npz \
  --output runs/hk/seed_0 \
  --seed 0 \
  --epochs 100 \
  --input-length 12 \
  --horizon 12 \
  --target-channel 0 \
  --batch-size 16 \
  --hidden 64 \
  --latent 16 \
  --components 3 \
  --alpha 1.75 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --forecast-weight 1.0 \
  --reconstruction-weight 0.1 \
  --renyi-weight 0.00001 \
  --renyi-ramp-epochs 10 \
  --em-warmup 5 \
  --em-variance-floor 0.5 \
  --em-max-points 100000 \
  --patience 15 \
  2>&1 | tee logs/hk_seed_0.log
```

At five-minute resolution, `--input-length 12` uses the previous hour and
`--horizon 12` predicts the following hour. Target channels are:

```text
--target-channel 0   # speed forecasting
--target-channel 1   # traffic-volume forecasting
--target-channel 2   # occupancy forecasting
```

Each run creates:

```text
runs/hk/seed_0/
├── history.csv
├── metrics.json
├── model.pt
└── predictions.npz
```

`predictions.npz` contains the test targets, model predictions, and persistence
predictions in the original physical scale. A positive value of
`mae_improvement_over_persistence_percent` means that the model outperforms the
persistence baseline.

## Ten-seed evaluation

The reported experiments should be repeated using the predetermined seeds
0--9. Do not select or discard runs based on their performance.

```bash
mkdir -p runs/hk logs

for seed in $(seq 0 9)
do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py \
    --data data/hong_kong/hk_traffic_20250601_20250621.npz \
    --output "runs/hk/seed_${seed}" \
    --seed "${seed}" \
    --epochs 100 \
    --input-length 12 \
    --horizon 12 \
    --target-channel 0 \
    --batch-size 16 \
    --hidden 64 \
    --latent 16 \
    --components 3 \
    --alpha 1.75 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --forecast-weight 1.0 \
    --reconstruction-weight 0.1 \
    --renyi-weight 0.00001 \
    --renyi-ramp-epochs 10 \
    --em-warmup 5 \
    --em-variance-floor 0.5 \
    --em-max-points 100000 \
    --patience 15 \
    2>&1 | tee "logs/hk_seed_${seed}.log"

  test ${PIPESTATUS[0]} -eq 0 || exit 1
done
```

Report the arithmetic mean and sample standard deviation across all ten runs.
Preserve every configuration, log, checkpoint, and result file used to produce
the manuscript tables.

## Synthetic integration test

The synthetic dataset is generated locally and is used only to verify that the
software pipeline executes correctly:

```bash
python scripts/make_synthetic_data.py

PYTHONPATH=src python scripts/train.py \
  --data work/synthetic_traffic.npz \
  --output work/synthetic_run \
  --epochs 8 \
  --input-length 12 \
  --horizon 3 \
  --em-warmup 2 \
  --em-variance-floor 0.5
```

This command creates `work/synthetic_traffic.npz`. The file is not downloaded
and does not need to be committed or uploaded. It is a software integration
fixture, not a research dataset, and results obtained from it must not be
reported as manuscript experiments.

## Reconstruct the dataset from official raw records

Researchers who wish to reconstruct the processed dataset from the government
archive can use:

```bash
python -u scripts/download_hk_history.py \
  --start 20250601 \
  --end 20250621 \
  --output data/hong_kong/raw_20250601_20250621

python -u scripts/prepare_hk_xml.py \
  --input data/hong_kong/raw_20250601_20250621 \
  --output data/hong_kong/hk_traffic_20250601_20250621.npz \
  --frequency-minutes 5 \
  --min-coverage 0.50 \
  --graph-k 3
```

The Historical Archive API can contain many snapshots, so downloading the full
period may take several hours. The downloader is resumable and records version
responses, failures, filenames, and checksums. See
[`HK_DATA_DOWNLOAD_README.md`](HK_DATA_DOWNLOAD_README.md) for the complete
download, preprocessing, validation, and troubleshooting instructions.

## Numerical validity condition

For `alpha > 1`, the closed-form Gaussian Rényi divergence is finite only when

```text
alpha / q_variance + (1-alpha) / p_variance > 0
```

in every latent dimension. The encoder caps the posterior variance at one, and
EM uses a configurable prior-variance floor. With the default `alpha=1.75`, the
prior variance floor must be greater than

```text
(alpha - 1) / alpha = 0.428571...
```

The default `--em-variance-floor 0.5` satisfies this condition. The training
program raises an explicit error if an invalid configuration is supplied rather
than silently clipping a non-finite divergence.

## Reproducibility checklist

Before reporting results:

1. verify that the Zenodo checksum matches;
2. keep the chronological 70%/20%/10% split unchanged;
3. fit imputation and normalization statistics on training data only;
4. estimate the graph and GMM prior from training data only;
5. run all predetermined seeds 0--9;
6. retain failed and low-performing runs rather than selectively removing them;
7. compute manuscript statistics from the saved result files;
8. archive the exact configuration, code version, logs, and predictions.

See [`EXPERIMENT_CHECKLIST.md`](EXPERIMENT_CHECKLIST.md) for additional checks.

## Citation

If you use the published Hong Kong dataset, please cite:

```bibtex
@dataset{zhang2026hktraffic,
  author    = {Zhang, Zheng and Yuan, Qiuyue and Zhang, Kaiyuan and Luo, Rui},
  title     = {HK-Traffic-GMVAE: Five-Minute Traffic Speed, Volume, and
               Occupancy Data from Hong Kong, June 2025},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22195838},
  url       = {https://doi.org/10.5281/zenodo.22195838}
}
```

## Data attribution and license

The underlying traffic observations remain attributable to:

> Transport Department, The Government of the Hong Kong Special
> Administrative Region, via DATA.GOV.HK.

The Zenodo record contains the applicable attribution, source URLs, quality
report, and reuse information. Users should consult those materials and the
current DATA.GOV.HK Terms and Conditions before redistributing raw government
records.

## Troubleshooting

### `ModuleNotFoundError: No module named 'gmvae'`

Run training from the repository root with `PYTHONPATH=src`, or install the
package in editable mode:

```bash
python -m pip install -e '.[dev]'
```

### `FileNotFoundError` for the Hong Kong NPZ

Download it first:

```bash
python scripts/download_zenodo_dataset.py
```

Then confirm the path:

```bash
ls -lh data/hong_kong/hk_traffic_20250601_20250621.npz
```

### `FileNotFoundError` for `work/synthetic_traffic.npz`

Generate the synthetic integration fixture:

```bash
python scripts/make_synthetic_data.py
```

For manuscript experiments, use the published Hong Kong NPZ instead.

### CUDA is unavailable

Check the driver and PyTorch installation:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU execution is supported but will be substantially slower for the complete
dataset.

## Acknowledgement

We acknowledge the Transport Department of the Government of the Hong Kong
Special Administrative Region and DATA.GOV.HK for providing the underlying
traffic-detector observations.

