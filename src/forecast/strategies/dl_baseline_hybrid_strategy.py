"""
Article-faithful Hybrid Transformer-LSTM reproducing Hussain et al. (2025).

    Hussain, A., Eswarakrishnan, V., Aslam, A. & Tripura, S.
    "Charging stations demand forecasting using LSTM based hybrid transformer model."
    Sci Rep 15, 13555 (2025). https://doi.org/10.1038/s41598-025-20421-y

Architecture (from paper Fig. 4 and Section "Proposed hybrid transformer model"):
  **Encoder:**
    Input → LSTM(d_model, return_sequences)
          → Sinusoidal Positional Encoding
          → MultiHeadAttention(dropout)
          → LayerNormalization
          → Dropout

  **Decoder:**
    encoder_output → LSTM(d_model, return_sequences)
                   → Sinusoidal Positional Encoding
                   → MultiHeadAttention(dropout)
                   → LayerNormalization
                   → Dropout

  **Head:**
    Flatten → Dense(1)

Key differences from our standard ``HybridTransformerLSTMModelStrategy``:
  * **No residual connections** — article Fig. 4 does not show Add() skip-paths.
  * **Explicit Dropout layers** after LayerNorm (article: "dropout for regularization").
  * **Sinusoidal Positional Encoding** between LSTM and MHA
    (article: "positional encoding is used to keep track of the order").
  * Default dropout = 0.2 (vs 0.1 in our standard variant).

Our ``HybridTransformerLSTMModelStrategy`` retains residual connections
following the canonical Transformer design (Vaswani et al., 2017).  This
class exists solely for paper-faithful comparison with Hussain et al.

Article hyperparameters (Table 1):
  epochs=100, batch_size=32, learning_rate=0.001,
  dropout=0.2, activation=ReLU, scaler=MinMaxScaler.
  look_back = forecast_horizon (30, 120, or 240 days).

Implementation notes — deviations and justified interpretations
---------------------------------------------------------------
1. **Decoder attention is self-attention, not cross-attention.**
   The paper text states the decoder attention "attends to the relevant
   encoded input", implying cross-attention (Q=dec, KV=enc_out).
   However, the decoder LSTM input IS enc_out, so cross-attention would be
   Q=LSTM(enc_out), KV=enc_out — querying back to the same signal just
   processed (circular, zero new information).  Self-attention on the
   decoded sequence is more principled and is our justified interpretation
   of the ambiguous Fig. 4.

2. **key_dim = d_model // num_heads** — the paper (Table 1) does not specify
   ``key_dim``.  We use the Vaswani (2017) standard: ``key_dim = d_model //
   num_heads = 32``.  This gives ~65k params per MHA layer, keeping total
   model size at ~200k (vs ~730k with the naive ``key_dim=d_model=128``).

3. **look_back** — ``max(MIN_LOOK_BACK=14, forecast_horizon)`` equals
   ``forecast_horizon`` for all tested horizons (30, 120d), matching the
   article's specification exactly.

4. **Prediction mode** — the article evaluates 30/120/240-day recursive
   (schedule) forecasts from the train/test split boundary.  Our code uses
   backtest (rolling one-step-ahead) mode for a fair comparison across all
   models under the same evaluation framework.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Flatten, Dropout, Lambda,
    LSTM as KerasLSTM,
    MultiHeadAttention, LayerNormalization,
)
from sklearn.preprocessing import MinMaxScaler

from .base_keras import KerasTimeSeriesBaseStrategy


def _sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    """Return a (seq_len, d_model) sinusoidal positional-encoding matrix.

    Follows the formulation from Vaswani et al. (2017), Eq. 3.
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10_000.0, (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates
    pe = np.zeros_like(angle_rads)
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pe.astype(np.float32)


class HussainHybridModelStrategy(KerasTimeSeriesBaseStrategy):
    """Article-faithful LSTM-Transformer Hybrid from Hussain et al. (2025).

    This strategy reproduces Fig. 4 of the paper **without** residual
    connections, adding positional encoding and explicit Dropout layers
    as described in the text.
    """

    @property
    def _params_key(self) -> str:
        return 'dl_baseline_hybrid_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'Hussain Hybrid'

    @property
    def _scaler_class(self):
        """Article Table 1: scaler=MinMaxScaler."""
        return MinMaxScaler

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        output_steps: int = 1,
        **_kwargs,
    ):
        """Build the article-faithful Hybrid encoder-decoder model.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, 1)`` for univariate data.
        d_model : int
            Hidden units for LSTM layers and attention key dimension.
        num_heads : int
            Number of parallel attention heads.
        dropout : float
            Dropout rate applied in attention and after LayerNorm.
        learning_rate : float
            Adam optimiser learning rate.
        """
        seq_len = input_shape[0]
        pe_matrix = _sinusoidal_pe(seq_len, d_model)

        inputs = Input(shape=input_shape)

        # --- Encoder (Fig. 4) ---
        # LSTM captures local temporal dependencies
        lstm_enc = KerasLSTM(units=d_model, return_sequences=True)(inputs)
        # Positional encoding preserves ordering for the attention layer
        lstm_enc_pe = Lambda(
            lambda x, _pe=pe_matrix: x + _pe,
            name='pe_encoder',
        )(lstm_enc)
        # Self-attention over encoder sequence.
        # key_dim = d_model // num_heads (Vaswani 2017 standard: 128//4=32).
        # The paper (Table 1) does not specify key_dim; using the standard
        # value prevents the 730k-param over-parameterisation of the naive
        # key_dim=d_model=128 choice.
        key_dim = max(1, d_model // num_heads)
        attn_enc = MultiHeadAttention(
            key_dim=key_dim, num_heads=num_heads, dropout=dropout,
        )(lstm_enc_pe, lstm_enc_pe)
        # NOTE: No residual Add — article Fig. 4 does not include skip connections
        enc_norm = LayerNormalization(epsilon=1e-6)(attn_enc)
        enc_out = Dropout(dropout)(enc_norm)

        # --- Decoder (Fig. 4) ---
        lstm_dec = KerasLSTM(units=d_model, return_sequences=True)(enc_out)
        lstm_dec_pe = Lambda(
            lambda x, _pe=pe_matrix: x + _pe,
            name='pe_decoder',
        )(lstm_dec)
        attn_dec = MultiHeadAttention(
            key_dim=key_dim, num_heads=num_heads, dropout=dropout,
        )(lstm_dec_pe, lstm_dec_pe)
        dec_norm = LayerNormalization(epsilon=1e-6)(attn_dec)
        dec_out = Dropout(dropout)(dec_norm)

        # --- Regression head ---
        flat = Flatten()(dec_out)
        outputs = Dense(output_steps)(flat)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model
