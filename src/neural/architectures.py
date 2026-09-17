"""
architectures.py - Neural network architectures for clinical abstract classification.

Models implemented:
1. TFIDF_MLP: Multilayer Perceptron on TF-IDF features with Dropout, BatchNorm, and weight decay.
2. Embedding_MLP: Trainable word embeddings + Global Average/Max Pooling + Dense layers.
3. VanillaRNN: Basic Recurrent Neural Network (many-to-one) with embedding layer.
4. BidirectionalLSTM: 2-layer Bidirectional LSTM with dynamic sequence masking.
5. AttentionBiLSTM: Bidirectional LSTM with additive self-attention mechanism.
"""

import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 1. MLP Baselines
# ============================================================================

class TFIDF_MLP(nn.Module):
    """
    Multilayer Perceptron operating on sparse/dense TF-IDF features.
    Architecture: Input -> Dense(256) -> BatchNorm -> ReLU -> Dropout(0.4) -> Dense(64) -> ReLU -> Dropout(0.3) -> Dense(1)
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 256, hidden_dim2: int = 64, dropout_rate: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.drop2 = nn.Dropout(dropout_rate * 0.75)

        self.out = nn.Linear(hidden_dim2, 1)

    def forward(self, x, lengths=None):
        h = self.drop1(F.relu(self.bn1(self.fc1(x))))
        h = self.drop2(F.relu(self.bn2(self.fc2(h))))
        logits = self.out(h).squeeze(-1)
        return logits


class Embedding_MLP(nn.Module):
    """
    Trainable Word Embeddings with Global Average Pooling (DAN - Deep Averaging Network).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, pad_idx: int = 0, dropout_rate: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop_embed = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embeds = self.drop_embed(self.embedding(x))  # [batch_size, seq_len, embed_dim]

        if lengths is not None:
            # Mask out padding tokens from average
            mask = (x != 0).unsqueeze(-1).float()
            sum_embeds = torch.sum(embeds * mask, dim=1)
            lens = lengths.unsqueeze(-1).clamp(min=1).float()
            pooled = sum_embeds / lens
        else:
            pooled = torch.mean(embeds, dim=1)

        h = self.drop1(F.relu(self.fc1(pooled)))
        logits = self.out(h).squeeze(-1)
        return logits


# ============================================================================
# 2. Vanilla RNN
# ============================================================================

class VanillaRNN(nn.Module):
    """
    Standard Elman Recurrent Neural Network (many-to-one).
    Extracts the hidden state at the last valid timestep for each sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        pad_idx: int = 0,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop_embed = nn.Dropout(0.2)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, 1)

    def forward(self, x, lengths):
        # x: [batch_size, seq_len], lengths: [batch_size]
        embeds = self.drop_embed(self.embedding(x))
        out, _ = self.rnn(embeds)  # out: [batch_size, seq_len, hidden_dim]

        # Gather last valid hidden state based on sequence length
        batch_size = x.size(0)
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(batch_size, 1, out.size(2))
        last_hidden = out.gather(1, idx).squeeze(1)  # [batch_size, hidden_dim]

        h = self.dropout(F.relu(self.fc(last_hidden)))
        logits = self.out(h).squeeze(-1)
        return logits


# ============================================================================
# 3. Bidirectional LSTM
# ============================================================================

class BidirectionalLSTM(nn.Module):
    """
    2-Layer Bidirectional LSTM with dynamic sequence masking and forward/backward state pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pad_idx: int = 0,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop_embed = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        # 2 * hidden_dim for bidirectional concatenation
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, 1)

    def forward(self, x, lengths):
        embeds = self.drop_embed(self.embedding(x))

        # Dynamic pack padded sequence for variable lengths
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Global max pooling across valid timesteps to capture salient clinical indicators
        mask = (x != 0).unsqueeze(-1).float()
        out_masked = out * mask + (1.0 - mask) * -1e9
        pooled, _ = torch.max(out_masked, dim=1)  # [batch_size, hidden_dim * 2]

        h = self.dropout(F.relu(self.fc(pooled)))
        logits = self.out(h).squeeze(-1)
        return logits


# ============================================================================
# 4. Attention-Based BiLSTM
# ============================================================================

class AdditiveAttention(nn.Module):
    """
    Additive Self-Attention mechanism over LSTM hidden states:
    u_t = tanh(W * h_t + b)
    alpha_t = softmax(u_t^T * v)
    context = sum(alpha_t * h_t)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, mask=None):
        # h: [batch_size, seq_len, hidden_dim]
        # u: [batch_size, seq_len, attn_dim]
        u = torch.tanh(self.W(h))
        # scores: [batch_size, seq_len]
        scores = self.v(u).squeeze(-1)

        if mask is not None:
            # Mask out padding positions with -1e9
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # [batch_size, hidden_dim]
        return context, attn_weights


class AttentionBiLSTM(nn.Module):
    """
    Bidirectional LSTM with Additive Self-Attention.
    Allows the model to dynamically weight crucial clinical markers (e.g. PSA thresholds, risk factors).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        pad_idx: int = 0,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.drop_embed = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_dim=hidden_dim * 2, attn_dim=64)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, 1)

    def forward(self, x, lengths, return_attention: bool = False):
        embeds = self.drop_embed(self.embedding(x))
        out, _ = self.lstm(embeds)  # out: [batch_size, seq_len, hidden_dim * 2]

        mask = (x != 0)  # [batch_size, seq_len]
        context, attn_weights = self.attention(out, mask=mask)

        h = self.dropout(F.relu(self.fc(context)))
        logits = self.out(h).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    print("Testing Neural Network Architectures...")
    batch_size = 4
    seq_len = 50
    vocab_size = 1000
    dummy_x = torch.randint(1, vocab_size, (batch_size, seq_len))
    dummy_lengths = torch.tensor([50, 40, 30, 20])
    dummy_tfidf = torch.randn(batch_size, 3000)

    # Test TFIDF_MLP
    mlp = TFIDF_MLP(input_dim=3000)
    out_mlp = mlp(dummy_tfidf)
    print(f"[OK] TFIDF_MLP: Output={out_mlp.shape}, Params={count_parameters(mlp):,}")

    # Test VanillaRNN
    rnn = VanillaRNN(vocab_size=vocab_size)
    out_rnn = rnn(dummy_x, dummy_lengths)
    print(f"[OK] VanillaRNN: Output={out_rnn.shape}, Params={count_parameters(rnn):,}")

    # Test BidirectionalLSTM
    bilstm = BidirectionalLSTM(vocab_size=vocab_size)
    out_lstm = bilstm(dummy_x, dummy_lengths)
    print(f"[OK] BidirectionalLSTM: Output={out_lstm.shape}, Params={count_parameters(bilstm):,}")

    # Test AttentionBiLSTM
    attn_lstm = AttentionBiLSTM(vocab_size=vocab_size)
    out_attn, weights = attn_lstm(dummy_x, dummy_lengths, return_attention=True)
    print(f"[OK] AttentionBiLSTM: Output={out_attn.shape}, AttnWeights={weights.shape}, Params={count_parameters(attn_lstm):,}")
