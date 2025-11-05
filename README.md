# Paper Reproduction

This repository contains the code to reproduce the results of the paper.

## Setup

To reproduce the results, you need to have conda installed.
You can create the conda environment with the following command:

```bash
conda env create -f environment.yaml
```

This will create a conda environment named `benchmark` with all the necessary packages.
Additionally, you need a Weights & Biases (wandb) account. The secltion below explains a minimal configuration needed to run the `code/hyper_tuning.py` and `code/independent_evaluation.py` script

### Weights & Biases Setup

- Create or use an existing wandb account at https://wandb.ai and copy your API key (Profile → Settings → API Keys).
- Activate the `benchmark` environment and authenticate by running `wandb login <your_api_key>` (or set `WANDB_API_KEY=<your_api_key>` in the shell before starting the script).
- By default runs are logged to the project named `hyper_tuning_v5` under your personal entity. Create that project in the wandb UI or change the `WANDB_PROJECT` constant in `code/hyper_tuning.py` to match an existing project. Optionally, set `WANDB_ENTITY=<team_or_username>` if you want to log to a specific team workspace.
- After authentication, `code/hyper_tuning.py` will create a run group per drug/target/model and store the local cache under `code/wandb/` automatically—no further setup is required.

## Hyperparameter Tuning

To reproduce the hyperparameter tuning, you can run the `hyper_tuning.py` script.
The script takes the following arguments:
- `--drugs`: A list of drugs to process.
- `--n_trials`: The number of Optuna trials.
- `--model`: The name of the model to tune.

For example, to run the hyperparameter tuning for the SCAD model on the drug Docetaxel with 10 Optuna trials, you can run the following command:

```bash
bash -c "conda activate benchmark && python code/hyper_tuning.py --drugs Docetaxel --n_trials 10 --model SCAD"
```

## Directory Structure

- `code/`: Contains the source code for the models and experiments.
- `datasets/`: Contains the datasets used in the paper.

## Code Overview

The `code/` folder gathers data processing helpers, experiment orchestration scripts, and the individual implementation of each transfer learning framework.

- `code/data_utils.py`: Shared data processing helpers for the harmonization of datasets, gene gene vocabularies (including scATD-specific alignment), etc. Also builds PyTorch dataloaders such as `CombinedDataLoader` for paired source/target batches and `create_shot_dataloaders` for semi-supervised setups.
- `code/training_utils.py`: Shared training helpers and baselines. It sets global seeds, defines callbacks (e.g., delayed early stopping), computes model metrics, and wraps framework-specific runners (`run_scad_benchmark`, `run_scdeal_benchmark`, etc.) alongside classical baselines such as CatBoost and RandomForest.
- `code/hyper_tuning.py`: Runs Optuna sweeps per drug/domain and logs trials to Weights & Biases. It standardizes preprocessing, constructs framework argument objects, and dispatches to the runners above.
- `code/independent_evaluation.py`: Repeats the preprocessing pipeline for held-out target datasets and launches framework benchmarks/few-shot baselines with consistent defaults, enabling cross-dataset comparisons.
- `code/frameworks/`: Houses the Lightning implementations of each domain adaptation method:
  - `SCAD/`: Domain-adversarial Lightning module that couples a shared encoder, response predictor, and gradient-reversal discriminator with a tunable weight lambda.
  - `scATD/`: Wraps a pre-trained Dist-VAE encoder and classifier head. `setup` loads checkpoints, aligning gene vocabularies by padding/truncation; fine-tuning alternates between frozen-classifier warm-up and optional encoder unfreezing, optimizing cross-entropy plus an RBF MMD penalty via manual optimization.
  - `scDeal/`: Implements the three-stage scDEAL workflow. Autoencoder/predictor pretraining is followed by a DaNN domain adaptation step with BCE, MMD, and Louvain-cluster similarity regularizers, orchestrated through manual optimization. Utilities in `scDEAL_utils.py` construct target KNN graphs and Louvain assignments.
  - `SSDA4Drug/`: Lightning module that implements, SSDA4Drug, with a shared encoder and classifier to which adversarial perturbations can be applied optionally. Training mixes supervised cross-entropy (source + few-shot target) with alternating entropy minimization and maximization on unlabeled target batches via `utils.adentropy`.

## Downloading Data
- To reproduce the results, you will need to download the datasesets used in the paper into  `datasets/processed/` [URL to be added].
- To run scATD, the pre-trained model weights (file checkpoint_fold1_epoch_30.pth) need to be downloaded from figshare (https://figshare.com/articles/software/scATD/27908847) and placed in `code/frameworks/scATD/pretrained_models/`.

    
