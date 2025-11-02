# Domain-Adaptation Benchmarks

This repository hosts modular PyTorch/PyTorch-Lightning implementations for several single-cell domain-adaptation frameworks:

- **SCAD** – Lightning module pairing a shared encoder/predictor with an adversarial discriminator and UMAP logging callbacks.
- **SSDA4Drug** – Semi-supervised pipeline with labeled/unlabeled combined loaders and entropy minimisation.
- **scDEAL** – Four-stage workflow (bulk AE, predictor, single-cell AE, DaNN) with extensive training utilities.
- **scATD** – Distillation-based classifier that aligns inputs to a pretrained scATD vocabulary and runs staged fine-tuning.
- **scADA** – Custom trainer orchestrating multi-domain source/target loops.

## Repository Layout

```
code/
├─ data_utils.py          # Data loading, alignment, batch sampling, symbol conversions
├─ model_utils.py         # Shared math utilities (MMD, Louvain clustering, adentropy)
├─ training_utils.py      # Metric helpers, benchmark entry points, experiment logging
├─ frameworks/            # Individual framework modules (SCAD, SSDA4Drug, scATD, scDEAL, scADA)
├─ hyperparameter_tuning/ # Optuna sweeps and single-model scripts
└─ base_trainer.py        # Generic training loop utilities shared by scADA
```

## Getting Started

1. **Install dependencies** – ensure the `SCAD` conda env (or equivalent) contains PyTorch, PyTorch Lightning, scikit-learn, CatBoost, wandb, and UMAP.
2. **Prepare data** – processed CSV splits should live under `datasets/processed`. Symbol/Gene vocab resources are referenced from `datasets/reference`.
3. **Run benchmark** – e.g. `python hyperparameter_tuning/hyper_tuning.py --drugs Vorinostat --n_trials 20`.
4. **Evaluate single model** – adjust `MODEL_NAME` and hyperparameters inside `hyperparameter_tuning/train_single_model.py` then execute the script.

## Notes

- Experiment results are logged via Weights & Biases; the utilities automatically initialise/save JSON summaries.
- Large pretrained weights (e.g., scATD Dist-VAE checkpoints) are excluded via `.gitignore` and must be placed manually under `frameworks/scATD/pretrained_models/`.
- The refactor eliminates the monolithic `utils.py`; all consumers now import from `data_utils`, `training_utils`, or `model_utils` as appropriate.

For questions or contributions, open an issue or reach out to the maintainers.
