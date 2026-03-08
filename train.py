"""
Entry point for training the CMR diagnostic classifier.

Usage:
    python train.py --config config/config.yaml --variant content
    python train.py --config config/config.yaml --variant conclusion
    python train.py --config config/config.yaml --variant both
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from src.dataset import make_data_loader
from src.model import DiagnosticClassifier
from src.train import run_training
from src.utils import build_pos_weight, set_seed, split_data


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, lr: float, weight_decay: float):
    """AdamW with separate weight-decay groups (no decay on bias / LayerNorm)."""
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=lr, correct_bias=True)


def train_variant(cfg: dict, variant: str, device: torch.device) -> None:
    """
    Train one model variant ("content" or "conclusion") end-to-end.

    Steps:
        1. Load and split the dataset.
        2. Build weighted loss (BCEWithLogitsLoss with pos_weight).
        3. Initialise model, optimiser, and scheduler.
        4. Run training loop with best-checkpoint saving.
    """
    vcfg = cfg[f"{variant}_model"]
    dcfg = cfg["data"]
    mcfg = cfg["model"]

    set_seed(dcfg["random_seed"])

    csv_key = f"{variant}_csv"
    text_col = f"{variant}_text_column"
    df = pd.read_csv(dcfg[csv_key])
    label_names = dcfg["label_names"]

    df_train, df_val, _ = split_data(
        df,
        random_seed=dcfg["random_seed"],
        stratify_col=dcfg["stratify_column"],
    )

    tokenizer = AutoTokenizer.from_pretrained(mcfg["pretrained_name"])

    loader_kwargs = dict(
        text_column=dcfg[text_col],
        label_names=label_names,
        tokenizer=tokenizer,
        max_len=vcfg["max_len"],
        batch_size=vcfg["batch_size"],
    )
    train_loader = make_data_loader(df_train, shuffle=True, **loader_kwargs)
    val_loader = make_data_loader(df_val, shuffle=False, **loader_kwargs)

    pos_weight = build_pos_weight(df_train[label_names].to_numpy(), device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = DiagnosticClassifier(
        n_classes=len(label_names),
        dropout=mcfg["dropout"],
        pretrained_name=mcfg["pretrained_name"],
    ).to(device)

    optimizer = build_optimizer(model, lr=vcfg["learning_rate"], weight_decay=vcfg["weight_decay"])
    total_steps = len(train_loader) * vcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=vcfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    os.makedirs(os.path.dirname(vcfg["checkpoint_path"]), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training variant: {variant.upper()}")
    print(f"  Epochs: {vcfg['epochs']}  |  Batch: {vcfg['batch_size']}  |  Max len: {vcfg['max_len']}")
    print(f"  Checkpoint: {vcfg['checkpoint_path']}")
    print(f"{'='*60}\n")

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=vcfg["epochs"],
        checkpoint_path=vcfg["checkpoint_path"],
    )


def main():
    parser = argparse.ArgumentParser(description="Train CMR diagnostic classifier")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--variant",
        choices=["content", "conclusion", "both"],
        default="both",
        help="Which model variant to train",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    variants = ["content", "conclusion"] if args.variant == "both" else [args.variant]
    for v in variants:
        train_variant(cfg, v, device)


if __name__ == "__main__":
    main()
