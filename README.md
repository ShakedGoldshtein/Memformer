# Memristor Language Model – WikiText-2

Train a transformer on WikiText-2 with memristor simulation (quantization, write noise, digital gradient storage).

## Setup

1. **Environment** – Miniconda virtual env
2. **Data** – WikiText-2 (loaded automatically from Hugging Face)
3. **Baseline** – Small transformer + training loop on WikiText-2
4. **Memristors** – Add quantization, noise, and gradient storage

## Run (Miniconda)

```bash
conda create -n guri python=3.11 -y
conda activate guri
pip install -r requirements.txt
python load_data.py   # verify data
python train.py
```

On each new terminal: `conda activate guri`
