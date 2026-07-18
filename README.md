# Extending CoFinDiff: Incorporating Macroeconomic Conditioning in Financial Time Series Generation

- **Undergraduate Researcher:** Andrew Liu
- **Institution:** University of California, Irvine
- **Faculty Mentor:** Dr. Weining Shen
- **Completed:** Spring 2026
- 📄 **[Full thesis here](./thesis.pdf)**

## Abstract

Financial data is limited and finite, yet modern financial modeling increasingly requires large
amounts of high-quality data for stress testing, algorithm validation, and scenario analysis. This need is
especially amplified by recent advancements in deep learning, where model performance often depends
on access to large training datasets. CoFinDiff addresses this challenge by using a diffusion-based
approach to generate synthetic financial time series while allowing controllable generation through
micro-level conditionings such as trend and realized volatility. This controllability enables synthetic data
to be generated with targeted behaviors and statistical properties rather than relying purely on stochastic
variation. This project extends the CoFinDiff framework by incorporating macroeconomic conditioning,
allowing the model to generate financial time series that respond to market-level variables such as interest
rates and the volatility index. By adding macroeconomic context, this work aims to improve the
flexibility, interpretability, and regime-awareness of controllable synthetic financial data generation.

## Research Question

CoFinDiff injects only micro-level conditioning variables (trend, realized volatiltiy). This project asks whether adding macro-level conditions (interest rate, VIX) changes how well the model learns controllable generation under a richer conditioning set.

## Repository Structure

```
config/          Hyperparameters
denoiser/        U-Net models and building blocks
diffusion/       DDPM-based training and sampling
preprocessing/   Data preprocessing helpers
training/        Training and tuning helpers
evaluate/        Evalutation and metrics helpers
notebooks/       End-to-end workflows (prep → train → generate → eval)
data/            Raw, training, and generated samples
models/          Checkpoints (Git LFS)
```

The project is notebook-driven. Run notebooks from the `notebooks/` directory. 

Dependencies, API keys, Git LFS, and GPU access may be needed.

## Citation

The original CoFinDiff research paper by Tanaka et al.:

```bibtex
@inproceedings{tanaka2025cofindiff,
  title={CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation},
  author={Tanaka, Yuki and Hashimoto, Ryuji and Takayanagi, Takehiro and Piao, Zhe and Murayama, Yuri and Izumi, Kiyoshi},
  booktitle={Proceedings of IJCAI},
  year={2025},
  doi={10.24963/ijcai.2025/1040}
}
```
