"""
SciBERT-based multi-label classifier for CMR report diagnosis.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_cased"


class DiagnosticClassifier(nn.Module):
    """
    Fine-tuned SciBERT model for multi-label classification of CMR reports.

    Architecture:
        SciBERT [CLS] pooled output
        → Dropout(p)
        → Linear(hidden_size → n_classes)

    The output layer uses Xavier uniform initialisation.
    Sigmoid activation is applied at inference time (not during training,
    because BCEWithLogitsLoss applies it internally for numerical stability).
    """

    def __init__(self, n_classes: int, dropout: float = 0.5, pretrained_name: str = SCIBERT_MODEL_NAME):
        """
        Args:
            n_classes: number of diagnostic labels to predict.
            dropout: dropout probability applied after pooled BERT output.
            pretrained_name: HuggingFace model identifier for the encoder.
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token id tensor.
            attention_mask: (batch, seq_len) mask tensor (1 = real token, 0 = padding).

        Returns:
            logits: (batch, n_classes) — raw scores before sigmoid.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output          # [CLS] representation
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
