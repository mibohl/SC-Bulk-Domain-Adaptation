# General
- when executing python scripts, always use the conda environment SCAD unless instructed otherwise
- Slurm submissions require `sbatch`; requests must use `--mem-per-cpu` (node-level memory requests are rejected).
- Contacting the scheduler from the sandbox fails; run `sbatch`/`squeue` with escalated permissions when prompted.

# Transfer Learning Framework Implementations

## SCAD (`code/frameworks/SCAD`)
- Implements a Lightning module that couples a shared encoder, drug-response predictor, and gradient-reversal discriminator for domain adversarial training.
- `SCADDataModule` zips balanced source and target loaders so each training step sees paired batches; validation/testing stay source-only vs. both domains.
- Loss blends BCE classification with discriminator BCE scaled by `lam1`, and logs per-domain metrics during multi-dataloader testing.
- `modules.py` contains reusable building blocks (encoders, predictor, discriminator, denoising autoencoder, adaptive-weighted variant) used across experiments.

## scADA (`code/frameworks/scADA`)
- Extends a custom `BaseTrainer` to orchestrate multi-source domain adaptation with manual training/validation loops.
- Shared autoencoder and predictor handle three interleaved source domains plus the target; a gradient-reversal discriminator drives adversarial alignment.
- Feature-vector generators weight source/target encodings, while orthogonality and reconstruction losses regularize diversity; total loss mixes prediction, reconstruction, adversarial, and diversity terms.
- Latent collection helpers support UMAP visualization by reusing the encoder pathway.

## scATD (`code/frameworks/scATD`)
- Wraps a pre-trained Dist-VAE encoder inside a Lightning module with a classifier head; handles two-phase training (frozen classifier warm-up, optional encoder unfreezing).
- `setup` loads checkpoint weights, padding/truncating tensors when gene vocabularies differ so pretrained knowledge transfers despite input-dimension mismatch.
- Fine-tuning couples cross-entropy on source labels with an RBF MMD penalty between source/target latent embeddings; manual optimization enables staged control.
- Data module aligns bulk and SCC47 expression matrices to the original scATD gene vocabulary before creating source/target splits via `train_test_split`.

## scDEAL (`code/frameworks/scDeal`)
- Lightning module coordinates the original three-stage pretraining (bulk autoencoder, bulk predictor, single-cell autoencoder) before launching domain adaptation.
- Main training step runs a DaNN wrapper that shares the bulk predictor with a target encoder; total loss combines BCE, MMD, and a Louvain-cluster similarity regularizer derived from target latent KNN graphs.
- Manual optimization lets the workflow alternate between attack/restore cycles, while the data module exposes both paired loaders for adaptation and standalone loaders for the pretraining hooks.

## SSDA4Drug (`code/frameworks/SSDA4Drug`)
- Semi-supervised Lightning module supporting either DAE or MLP encoders, paired with a predictor and an adentropy head for entropy minimization.
- Training blends supervised cross-entropy (source + few target labels) with optional encoder reconstruction loss; Fast Gradient Method (FGM) adversarial steps perturb shared parameters when `method="adv"`.
- Unlabeled target batches trigger an adentropy regularizer via `utils.adentropy`, encouraging confident predictions without labels.
- Data module builds standard source loaders and `n`-shot target loaders (labeled/unlabeled) via `utils.create_shot_dataloaders`, keeping evaluation on separate source/target test iterators.
