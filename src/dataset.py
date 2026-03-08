"""
Dataset classes and DataLoader factory functions for CMR report classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class CMRDataset(Dataset):
    """
    PyTorch Dataset for Cardiac MRI report multi-label classification.

    Each sample tokenizes a text field (report Content or Conclusion) and
    returns the corresponding binary target vector.
    """

    def __init__(self, texts, targets, tokenizer, max_len: int):
        """
        Args:
            texts: iterable of raw report strings.
            targets: array-like of shape (n_samples, n_classes) with 0/1 labels.
            tokenizer: HuggingFace tokenizer (e.g. SciBERT).
            max_len: maximum token sequence length (tokens beyond this are truncated).
        """
        self.texts = list(texts)
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float),
        }


def make_data_loader(
    df,
    text_column: str,
    label_names: list,
    tokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader from a DataFrame slice.

    Args:
        df: pandas DataFrame containing the text column and label columns.
        text_column: name of the column to use as input text (e.g. "Content" or "Conclusion").
        label_names: ordered list of label column names.
        tokenizer: HuggingFace tokenizer.
        max_len: maximum token sequence length.
        batch_size: mini-batch size.
        shuffle: whether to shuffle samples each epoch.
        num_workers: number of DataLoader worker processes.

    Returns:
        A PyTorch DataLoader.
    """
    dataset = CMRDataset(
        texts=df[text_column].to_numpy(),
        targets=df[label_names].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
