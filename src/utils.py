"""
Utility functions: class-frequency weighting, gradient-based saliency,
data splitting, and seed initialisation.
"""

import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Class-frequency weighting (for imbalanced multi-label classification)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute positive and negative class frequencies for each label.

    These are used to build the pos_weight tensor for BCEWithLogitsLoss,
    helping address label imbalance in multi-label settings.

    Args:
        labels: binary array of shape (n_samples, n_classes).

    Returns:
        positive_frequencies: fraction of positive examples per class.
        negative_frequencies: fraction of negative examples per class.
    """
    n = labels.shape[0]
    pos_freq = labels.sum(axis=0) / n
    neg_freq = (1 - labels).sum(axis=0) / n
    return pos_freq, neg_freq


def build_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Build the pos_weight tensor for BCEWithLogitsLoss from training labels.

    pos_weight[i] = neg_freq[i] / pos_freq[i], clipped to avoid division by zero.

    Args:
        labels: binary array of shape (n_samples, n_classes).
        device: target torch.device.

    Returns:
        pos_weight tensor of shape (n_classes,).
    """
    pos_freq, neg_freq = compute_class_weights(labels)
    pos_weight = neg_freq / np.clip(pos_freq, 1e-6, None)
    return torch.tensor(pos_weight, dtype=torch.float).to(device)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_data(df, random_seed: int = 13, stratify_col: str = "DCM"):
    """
    Perform a reproducible 80/10/10 train/val/test split.

    A stratified split is first applied on `stratify_col` (default "DCM") to
    preserve class proportions, matching the original notebook methodology.

    Args:
        df: pandas DataFrame with all data.
        random_seed: random seed for reproducibility.
        stratify_col: column name to stratify on.

    Returns:
        (df_train, df_val, df_test) DataFrames.
    """
    df_train, df_test = train_test_split(
        df,
        test_size=0.20,
        random_state=random_seed,
        stratify=df[stratify_col],
    )
    df_val, df_test = train_test_split(
        df_test,
        test_size=0.50,
        random_state=random_seed,
        stratify=df_test[stratify_col],
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Gradient-based saliency / interpretability
# ---------------------------------------------------------------------------

def compute_token_saliency(model, tokenizer, text: str, device: torch.device) -> tuple[list[str], np.ndarray]:
    """
    Compute gradient-based saliency scores for each token in `text`.

    Method:
        1. Tokenize the input and embed it.
        2. Run a forward pass keeping the embedding activations in the graph.
        3. Sum all output logits and back-propagate.
        4. Saliency = L2 norm of the gradient w.r.t. each token embedding.

    This highlights which words/tokens the model attended to most strongly
    for its predictions, supporting result transparency.

    Args:
        model: DiagnosticClassifier instance (should be in eval mode).
        tokenizer: HuggingFace tokenizer.
        text: raw report string.
        device: torch.device.

    Returns:
        tokens: list of decoded token strings.
        saliency: 1-D numpy array of saliency scores, one per token.
    """
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get the embedding matrix and register a hook to capture gradients
    embedding_matrix = model.bert.embeddings.word_embeddings(input_ids)
    embedding_matrix.retain_grad()

    logits = model.bert(
        inputs_embeds=embedding_matrix,
        attention_mask=attention_mask,
    ).pooler_output
    logits = model.classifier(model.dropout(logits))

    # Back-propagate through the sum of logits
    logits.sum().backward()

    # Saliency = L2 norm across the embedding dimension
    grad = embedding_matrix.grad.detach().cpu().numpy()[0]  # (seq_len, hidden_size)
    saliency = np.linalg.norm(grad, axis=-1)                # (seq_len,)

    # Decode tokens (skip padding)
    token_ids = input_ids[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Trim to non-padding tokens
    n_real = int(attention_mask[0].sum().item())
    return tokens[:n_real], saliency[:n_real]
