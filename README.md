# Extending CoFinDiff

**Honors Thesis Research** — Andrew Liu

[Institution / Program]  
[Thesis Title]

This repository contains the code and experiments for an honors thesis that extends [CoFinDiff](https://doi.org/10.24963/ijcai.2025/1040) (Tanaka et al., IJCAI 2025), a conditional diffusion model for synthetic financial time series generation. The original model conditions generation on micro-level price statistics (trend and realized volatility) via cross-attention over a Haar wavelet 2D representation of log returns. This work adds macro-level conditioning (interest rate and VIX), compares two U-Net conditioning architectures, and provides a full pipeline for data preparation, training, generation, and evaluation.

## Research Question

CoFinDiff injects all conditioning variables through spatial cross-attention. This project asks whether separating macro conditions (interest rate, VIX) into a FiLM (Feature-wise Linear Modulation) pathway—while keeping micro conditions (trend, realized volatility) in cross-attention—changes how well the model learns controllable generation under a richer conditioning set.

## Contributions

| Contribution | Location |
|---|---|
| **CA model** — all four conditions via spatial cross-attention | `denoiser/unet_model_ca.py` |
| **CA-FiLM model** — micro via cross-attention, macro (IR, VIX) via FiLM after each attention block | `denoiser/unet_model_ca_film.py` |
| **Macro conditioning** beyond the original CoFinDiff setup (trend + realized vol only) | `config/preprocess_config.py`, `preprocessing/condition_encoder.py` |
| **Classifier-free guidance** (scale 7.0, 20% unconditional dropout during training) | `config/diffusion_config.py`, `config/training_config.py` |
| **Condition shuffling** during training (probability 0.5) to reduce spurious condition–sample pairings | `config/training_config.py` |
| **Upsampling** of high-trend and high-volatility windows for class balance | `notebooks/prep/preprocess.ipynb` |
| **Evaluation suite** — stylized facts, diversity (Euclidean and DTW), micro/macro conditioning adherence | `evaluate/`, `notebooks/eval/` |

## Architecture

Both models share the same diffusion backbone and data representation.

**Shared pipeline.** Daily log returns are sliced into 64-timestep windows, globally standardized, and transformed into 8×8 images via a 6-level Haar wavelet decomposition. A DDPM with 1000 timesteps, linear beta schedule, and noise-prediction objective trains a U-Net denoiser on these images.

**Conditioning variables.**

- *Micro:* trend (scaled cumulative return) and realized volatility (scaled sum of squared returns)
- *Macro:* federal funds rate and VIX at the window start date

**CA model.** All four conditioning scalars are encoded and injected as context tokens through spatial cross-attention at each U-Net encoder level, the bottleneck, and each decoder level.

**CA-FiLM model.** Micro conditions (trend, realized vol) follow the same cross-attention pathway. Macro conditions (interest rate, VIX) are passed through FiLM layers that apply per-channel scale and shift after each cross-attention block.

**Model hyperparameters** (from `config/denoiser_config.py` and `config/diffusion_config.py`):

- 3-level U-Net with `BASE_CHANNELS=64`, `CHANNEL_MULT=[1, 2, 4, 8]`, 3 residual blocks per level
- Cross-attention: 4 heads, context dimension 32
- Training: batch size 32, learning rate 1.4e-6, up to 3000 epochs, early stopping patience 100

## Data

- **Universe:** ~200 tickers (US large-caps and international equities), defined in `config/data_config.py`
- **Date range:** 2000-01-01 to 2019-12-31
- **Training set:** 17,014 windows in `data/train/train_data_2d.json`, split 80/20 into 13,611 train and 3,403 validation samples
- **Normalization:** global standard deviation scaling (`global_std` stored in `data/train/global_stats.json`)
- **Generated sample naming:** `t{trend}r{rv}i{ir*100}v{vix}.json` (see `evaluate/generated_io.py`)

Raw daily price JSON files for most tickers are already committed under `data/raw/`. Re-fetching requires Alpha Vantage API keys (see Setup below).

## Repository Structure

```
config/          Hyperparameters (data, denoiser, diffusion, training)
denoiser/        U-Net models (CA, CA-FiLM) and building blocks
diffusion/       DDPM training and sampling
preprocessing/   Log returns, Haar wavelet, condition encoders
training/        Dataset, train/val steps
evaluate/        Metrics (kurtosis, ACF, DTW, conditioning adherence)
notebooks/       End-to-end workflows (prep → train → generate → eval)
data/            Raw, preprocessed, training, and generated samples
models/          Checkpoints (Git LFS)
```

## Setup

### Environment

```bash
conda env create -f environment.yaml
conda activate research
git lfs install
git lfs pull   # required for model checkpoints in models/
```

### API Keys

Create a `.env` file at the project root (see `.env.example`):

```
AV_API_KEY=your_alpha_vantage_key
AV_API_KEY_PREMIUM=optional_premium_key
AV_BASE_URL=https://www.alphavantage.co/query
```

API keys are only needed when rebuilding raw data via `notebooks/prep/get_data.ipynb`. VIX data is stored at `data/raw/daily_vix.json`.

### Pretrained Models

Checkpoints are stored via Git LFS:

- `models/ca/checkpoints/best_model.pt`
- `models/ca-film/checkpoints/best_model.pt`

## Reproduction Workflow

The project is notebook-driven; there are no CLI entry points. Run notebooks from the `notebooks/` directory — each notebook adds the project root to `sys.path`.

| Step | Notebook | Notes |
|---|---|---|
| 1. Fetch data | `notebooks/prep/get_data.ipynb` | Alpha Vantage + VIX; skip if using committed `data/raw/` |
| 2. Preprocess | `notebooks/prep/preprocess.ipynb` | Builds `data/train/train_data_2d.json` |
| 3. Train | `notebooks/ca/train.ipynb` or `notebooks/ca-film/train.ipynb` | Logs to W&B project `Extending-CoFinDiff`; GPU recommended |
| 4. Sweep (optional) | `notebooks/ca/sweep.ipynb` or `notebooks/ca-film/sweep.ipynb` | Uses `sweep_config_ca.yaml` or `sweep_config_ca_film.yaml` |
| 5. Generate | `notebooks/ca/generate.ipynb` or `notebooks/ca-film/generate.ipynb` | Conditional sample grids → `data/generated_*/` |
| 6. Evaluate | `notebooks/eval/*.ipynb` | Stylized facts, diversity, micro/macro conditioning |

Training notebooks include Colab setup cells (`git clone` + `%cd`). For local use, open the notebook directly from this repository.

## Citation

If you use CoFinDiff, please cite the original paper:

```bibtex
@inproceedings{tanaka2025cofindiff,
  title={CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation},
  author={Tanaka, Yuki and Hashimoto, Ryuji and Takayanagi, Takehiro and Piao, Zhe and Murayama, Yuri and Izumi, Kiyoshi},
  booktitle={Proceedings of IJCAI},
  year={2025},
  doi={10.24963/ijcai.2025/1040}
}
```

This work:

```bibtex
@mastersthesis{liu2026extending,
  title={[Your Thesis Title]},
  author={Liu, Andrew},
  school={[Your Institution]},
  year={2026}
}
```

## Acknowledgments

Alpha Vantage (market data), yfinance (VIX), Weights & Biases (experiment tracking), dtaidistance (DTW metrics), PyTorch.
