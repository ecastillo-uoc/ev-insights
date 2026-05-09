"""
Hybrid Transformer-LSTM encoder-decoder strategy for time-series forecasting.

Architecture (Functional API)::

  **Encoder:**
    Input → LSTM(d_model, return_sequences) → SinusoidalPE(look_back, d_model) →
            MHA (self-attn) + residual Add + LayerNorm →
            FFN Dense(ff_dim, relu) → Dense(d_model) + residual Add + LayerNorm

  **Decoder:**
    encoder_output → LSTM(d_model, return_sequences) → SinusoidalPE(look_back, d_model) →
                     MHA (cross-attn: Q=dec, KV=enc) + residual Add + LayerNorm →
                     FFN Dense(ff_dim, relu) → Dense(d_model) + residual Add + LayerNorm

  **Head:**
    GlobalAveragePooling1D → Dense(ff_dim, relu) → Dense(output_steps)

Hyperparameters read from ``df.attrs['hybrid_params']``:
  epochs, batch_size, learning_rate, look_back,
  d_model, num_heads, ff_dim, dropout.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Dropout, Lambda, GlobalAveragePooling1D,
    LSTM as KerasLSTM,
    MultiHeadAttention, LayerNormalization, Add,
)

from .base_keras import KerasTimeSeriesBaseStrategy, _sinusoidal_pe


class HybridTransformerLSTMModelStrategy(KerasTimeSeriesBaseStrategy):
    """LSTM encoder-decoder with Transformer attention for univariate forecasting."""

    @property
    def _params_key(self) -> str:
        return 'hybrid_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'Hybrid Transformer-LSTM'

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        d_model: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        output_steps: int = 1,
        **_kwargs,
    ):
        """Build the hybrid encoder-decoder model.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, 1)`` for univariate data.
        d_model : int
            Hidden units for LSTM layers and attention key dimension.
        num_heads : int
            Number of parallel attention heads.
        ff_dim : int
            Hidden units in the FFN sub-block of each encoder/decoder block
            (Vaswani et al. 2017, §3.3).  Default 256 = 2 × d_model.
        dropout : float
            Dropout rate applied inside attention layers and FFN sub-blocks.
        learning_rate : float
            Adam optimiser learning rate.
        output_steps : int
            Number of output values.  1 for single-step; ``forecast_horizon``
            for direct multi-step models.
        """
        seq_len = input_shape[0]
        pe_matrix = _sinusoidal_pe(seq_len, d_model)  # (look_back, d_model)

        inputs = Input(shape=input_shape)

        # ── Encoder ──────────────────────────────────────────────────
        lstm_enc = KerasLSTM(units=d_model, return_sequences=True)(inputs)
        # Positional encoding parameterised by look_back (= input_shape[0])
        lstm_enc_pe = Lambda(
            lambda x, _pe=pe_matrix: x + _pe, name='pe_encoder',
        )(lstm_enc)
        # Self-attention + residual + LayerNorm
        attn_enc = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout,
        )(lstm_enc_pe, lstm_enc_pe)
        enc_res1 = Add()([attn_enc, lstm_enc_pe])
        enc_ln1 = LayerNormalization(epsilon=1e-6)(enc_res1)
        # Feed-forward sub-block (Vaswani et al. 2017, §3.3)
        enc_ff = Dense(ff_dim, activation='relu')(enc_ln1)
        enc_ff = Dropout(dropout)(enc_ff)
        enc_ff = Dense(d_model)(enc_ff)
        enc_out = LayerNormalization(epsilon=1e-6)(Add()([enc_ff, enc_ln1]))

        # ── Decoder ──────────────────────────────────────────────────
        lstm_dec = KerasLSTM(units=d_model, return_sequences=True)(enc_out)
        lstm_dec_pe = Lambda(
            lambda x, _pe=pe_matrix: x + _pe, name='pe_decoder',
        )(lstm_dec)
        # Cross-attention: Q from decoder, K/V from encoder (Vaswani et al. 2017, §3.2)
        attn_dec = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout,
        )(lstm_dec_pe, enc_out)
        dec_res1 = Add()([attn_dec, lstm_dec_pe])
        dec_ln1 = LayerNormalization(epsilon=1e-6)(dec_res1)
        # Feed-forward sub-block
        dec_ff = Dense(ff_dim, activation='relu')(dec_ln1)
        dec_ff = Dropout(dropout)(dec_ff)
        dec_ff = Dense(d_model)(dec_ff)
        dec_out = LayerNormalization(epsilon=1e-6)(Add()([dec_ff, dec_ln1]))

        # ── Regression head ────────────────────────────────────────────
        # Pool over time dimension (look_back) → (batch, d_model)
        x = GlobalAveragePooling1D()(dec_out)
        x = Dense(ff_dim, activation='relu')(x)
        outputs = Dense(output_steps)(x)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model
