"""
Hybrid Transformer-LSTM encoder-decoder strategy for time-series forecasting.

Architecture (Functional API):
  **Encoder:**
    Input → LSTM(d_model, return_sequences) →
            MultiHeadAttention + residual Add + LayerNorm

  **Decoder:**
    encoder_output → LSTM(d_model, return_sequences) →
                     MultiHeadAttention + residual Add + LayerNorm

  **Head:**
    Flatten → Dense(1)

Hyperparameters read from ``df.attrs['hybrid_params']``:
  epochs, batch_size, learning_rate, look_back,
  d_model, num_heads, dropout.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Flatten,
    LSTM as KerasLSTM,
    MultiHeadAttention, LayerNormalization, Add,
)

from .base_keras import KerasTimeSeriesBaseStrategy


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
        dropout: float = 0.1,
        learning_rate: float = 0.001,
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
        dropout : float
            Dropout rate applied in attention layers.
        learning_rate : float
            Adam optimiser learning rate.
        """
        inputs = Input(shape=input_shape)

        # Encoder: LSTM + self-attention + residual + LayerNorm
        lstm_enc = KerasLSTM(units=d_model, return_sequences=True)(inputs)
        attn_enc = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout,
        )(lstm_enc, lstm_enc)
        enc_add = Add()([attn_enc, lstm_enc])
        enc_out = LayerNormalization(epsilon=1e-6)(enc_add)

        # Decoder: LSTM + self-attention + residual + LayerNorm
        lstm_dec = KerasLSTM(units=d_model, return_sequences=True)(enc_out)
        attn_dec = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads, dropout=dropout,
        )(lstm_dec, lstm_dec)
        dec_add = Add()([attn_dec, lstm_dec])
        dec_out = LayerNormalization(epsilon=1e-6)(dec_add)

        # Regression head
        flat = Flatten()(dec_out)
        outputs = Dense(1)(flat)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
        return model
