# Server runbook: Hong Kong XML to EM-GMVAE

Assumed paths:

```text
Project: /home/kaiyuan/zz/HK-trackdata/GMVAE-revision-aux-renyi
XML:     /home/kaiyuan/zz/HK-trackdata/hk_traffic_raw/hk_traffic_raw
```

## 1. Environment

```bash
cd /home/kaiyuan/zz/HK-trackdata/GMVAE-revision-aux-renyi
conda create -n gmvae-revision python=3.11 -y
conda activate gmvae-revision
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch using the command appropriate for `nvidia-smi` from the official PyTorch selector, then:

```bash
pip install -e '.[dev]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q
```

## 2. Convert a 500-file subset first

```bash
mkdir -p data/hong_kong work runs logs
python scripts/prepare_hk_xml.py \
  --input /home/kaiyuan/zz/HK-trackdata/hk_traffic_raw/hk_traffic_raw \
  --output work/hk_smoke.npz \
  --frequency-minutes 5 --min-coverage 0.50 --graph-k 3 --max-files 500
```

## 3. Train the subset

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py \
  --data work/hk_smoke.npz --output runs/hk_smoke \
  --seed 0 --epochs 20 --input-length 12 --horizon 3 \
  --batch-size 4 --hidden 32 --latent 8 --components 3 --alpha 1.75 \
  --forecast-weight 1.0 --reconstruction-weight 0.1 --renyi-weight 0.00001 \
  --renyi-ramp-epochs 10 --em-warmup 5 --em-variance-floor 0.5 \
  --em-max-points 20000 --patience 8
```

## 4. Convert the complete month

```bash
tmux new -s hkprepare
conda activate gmvae-revision
cd /home/kaiyuan/zz/HK-trackdata/GMVAE-revision-aux-renyi
python scripts/prepare_hk_xml.py \
  --input /home/kaiyuan/zz/HK-trackdata/hk_traffic_raw/hk_traffic_raw \
  --output data/hong_kong/hk_aug2025_5min.npz \
  --frequency-minutes 5 --min-coverage 0.50 --graph-k 3 \
  2>&1 | tee logs/prepare_hk_aug2025.log
```

Detach with `Ctrl+B`, then `D`; reattach with `tmux attach -t hkprepare`.

## 5. Validate the complete NPZ

```bash
cat data/hong_kong/hk_aug2025_5min.report.json
python - <<'PY'
import numpy as np
d = np.load('data/hong_kong/hk_aug2025_5min.npz')
for key in d.files:
    print(key, d[key].shape, d[key].dtype)
v, a = d['values'], d['adjacency']
print('NaN ratio:', np.isnan(v).mean())
print('finite or NaN:', np.all(np.isfinite(v) | np.isnan(v)))
print('adjacency symmetric:', np.allclose(a, a.T))
print('undirected edges:', np.count_nonzero(np.triu(a, 1)))
PY
```

## 6. One-epoch real-data run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py \
  --data data/hong_kong/hk_aug2025_5min.npz --output runs/hk_real_smoke \
  --seed 0 --epochs 1 --input-length 12 --horizon 12 --target-channel 0 \
  --batch-size 2 --hidden 32 --latent 8 --components 3 --alpha 1.75 \
  --em-warmup 5 --em-variance-floor 0.5 --em-max-points 50000 --patience 5
```

## 7. Full seed 0 run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py \
  --data data/hong_kong/hk_aug2025_5min.npz --output runs/hk/seed_0 \
  --seed 0 --epochs 100 --input-length 12 --horizon 12 --target-channel 0 \
  --batch-size 8 --hidden 64 --latent 16 --components 3 --alpha 1.75 \
  --learning-rate 0.001 \
  --forecast-weight 1.0 --reconstruction-weight 0.1 --renyi-weight 0.00001 \
  --renyi-ramp-epochs 10 \
  --em-warmup 5 --em-variance-floor 0.5 --em-max-points 100000 --patience 15 \
  2>&1 | tee logs/hk_seed_0.log
```

Inspect:

```bash
cat runs/hk/seed_0/metrics.json
tail -20 runs/hk/seed_0/history.csv
```

Positive `mae_improvement_over_persistence_percent` means better than persistence. A negative value is not yet a defensible forecasting result.

## 8. Ten predetermined seeds

Only after seed 0 is finite and better than persistence:

```bash
for seed in $(seq 0 9)
do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py \
    --data data/hong_kong/hk_aug2025_5min.npz \
    --output "runs/hk/seed_${seed}" \
    --seed "${seed}" --epochs 100 \
    --input-length 12 --horizon 12 --target-channel 0 \
    --batch-size 8 --hidden 64 --latent 16 \
    --components 3 --alpha 1.75 --learning-rate 0.001 \
    --forecast-weight 1.0 --reconstruction-weight 0.1 --renyi-weight 0.00001 \
    --renyi-ramp-epochs 10 \
    --em-warmup 5 --em-variance-floor 0.5 --em-max-points 100000 --patience 15 \
    2>&1 | tee "logs/hk_seed_${seed}.log"
  test ${PIPESTATUS[0]} -eq 0 || exit 1
done
```

Do not delete low-performing runs. Manuscript claims must use all ten result files.
