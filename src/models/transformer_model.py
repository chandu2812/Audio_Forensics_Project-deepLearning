"""Transformer-based model for audio deepfake detection."""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AudioTransformer(nn.Module):
    """
    Transformer encoder for Mel spectrogram classification.

    Treats each Mel frequency bin as a token (or alternatively each time frame).
    Here we flatten the (n_mels, time) spectrogram into a sequence of tokens.
    """

    def __init__(
        self,
        n_mels: int = 128,
        target_frames: int = 256,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        # Project each time-frame vector (n_mels) to d_model
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=target_frames + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        x = x.squeeze(1)  # (batch, n_mels, time)
        x = x.permute(0, 2, 1)  # (batch, time, n_mels)
        x = self.input_proj(x)  # (batch, time, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, time, d_model)
        x = x.mean(dim=1)  # global average pooling over time
        return self.classifier(x)