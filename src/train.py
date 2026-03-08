"""
Training and validation loop functions for the CMR diagnostic classifier.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Gradient clipping (max_norm=1.0) is applied before each optimiser step
    to stabilise fine-tuning of the pre-trained encoder.

    Args:
        model: DiagnosticClassifier instance.
        data_loader: training DataLoader.
        loss_fn: BCEWithLogitsLoss (or compatible).
        optimizer: AdamW optimiser.
        device: torch.device.
        scheduler: learning-rate scheduler (stepped once per batch).

    Returns:
        (mean_f1_micro, mean_loss) over all batches.
    """
    model.train()
    losses = []
    f1_scores = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        labels = targets.cpu().numpy()
        f1_scores.append(f1_score(labels, preds, average="micro", zero_division=0))
        losses.append(loss.item())

    return float(np.mean(f1_scores)), float(np.mean(losses))


@torch.no_grad()
def eval_model(model, data_loader, loss_fn, device) -> tuple[float, float]:
    """
    Evaluate the model on a validation or test DataLoader.

    Args:
        model: DiagnosticClassifier instance.
        data_loader: validation/test DataLoader.
        loss_fn: BCEWithLogitsLoss (or compatible).
        device: torch.device.

    Returns:
        (mean_f1_micro, mean_loss) over all batches.
    """
    model.eval()
    losses = []
    f1_scores = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, targets)

        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        labels = targets.cpu().numpy()
        f1_scores.append(f1_score(labels, preds, average="micro", zero_division=0))
        losses.append(loss.item())

    return float(np.mean(f1_scores)), float(np.mean(losses))


@torch.no_grad()
def get_predictions(model, data_loader, device) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Collect predictions, probabilities, and raw texts from a DataLoader.

    Args:
        model: DiagnosticClassifier instance.
        data_loader: any DataLoader (typically test set).
        device: torch.device.

    Returns:
        predictions: (n_samples, n_classes) binary array.
        probabilities: (n_samples, n_classes) float array in [0, 1].
        texts: list of raw text strings.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_texts = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_preds.append(preds)
        all_probs.append(probs)
        all_texts.extend(batch["text"])

    return np.vstack(all_preds), np.vstack(all_probs), all_texts


def run_training(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    epochs: int,
    checkpoint_path: str,
) -> dict:
    """
    Full training loop with best-model checkpointing on validation F1.

    Args:
        model: DiagnosticClassifier.
        train_loader: training DataLoader.
        val_loader: validation DataLoader.
        loss_fn: BCEWithLogitsLoss.
        optimizer: AdamW optimiser.
        scheduler: linear warmup scheduler.
        device: torch.device.
        epochs: number of training epochs.
        checkpoint_path: path to save the best model state dict (.bin file).

    Returns:
        history dict with keys "train_f1", "val_f1", "train_loss", "val_loss".
    """
    history = {"train_f1": [], "val_f1": [], "train_loss": [], "val_loss": []}
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        train_f1, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        val_f1, val_loss = eval_model(model, val_loader, loss_fn, device)

        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train F1: {train_f1:.4f}  Loss: {train_loss:.4f} | "
            f"Val F1: {val_f1:.4f}  Loss: {val_loss:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → New best model saved (val F1 = {best_val_f1:.4f})")

    return history
