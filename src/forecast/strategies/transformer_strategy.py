"""
Transformer encoder strategy for time-series forecasting.

Architecture (Functional API):
  Input → [N × TransformerEncoder blocks] → GlobalAveragePooling1D →
          MLP head (Dense(128) + Dropout) → Dense(1)

Each TransformerEncoder block:
  LayerNorm → MultiHeadAttention → residual Add →
  LayerNorm → Dense(ff_dim) → Dropout → Dense(d_input) → residual Add

Hyperparameters read from ``df.attrs['transformer_params']``:
  epochs, batch_size, learning_rate, look_back,
  head_size, num_heads, ff_dim, num_transformer_blocks,
  mlp_units, dropout, mlp_dropout.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Dropout,
    MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D,
)

from .base_keras import KerasTimeSeriesBaseStrategy


class TransformerModelStrategy(KerasTimeSeriesBaseStrategy):
    """Multi-block Transformer encoder with MLP head for univariate forecasting."""

    @property
    def _params_key(self) -> str:
        return 'transformer_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'Transformer'

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        """Single Transformer encoder block with residual connections."""
        # Normalisation and Attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
        )(x, x)
        x = Dropout(dropout)(x)
        res = Add()([x, inputs])

        # Feed-forward part
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        return Add()([x, res])

    # ------------------------------------------------------------------
    # build_model
    # ------------------------------------------------------------------

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        head_size: int = 256,
        num_heads: int = 4,
        ff_dim: int = 4,
        num_transformer_blocks: int = 4,
        mlp_units: list | None = None,
        dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        learning_rate: float = 0.001,
        **_kwargs,
    ):
        """Construct a multi-block Transformer encoder model.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, 1)`` for univariate data.
        head_size : int
            Dimensionality of keys in each attention head.
        num_heads : int
            Number of parallel attention heads.
        ff_dim : int
            Hidden units in the point-wise feed-forward block.
        num_transformer_blocks : int
            Number of stacked encoder blocks.
        mlp_units : list[int]
            Widths of Dense layers in the MLP classification head.
        dropout, mlp_dropout : float
            Dropout rates for encoder and MLP head respectively.
        learning_rate : float
            Adam optimiser learning rate.
        """
        if mlp_units is None:
            mlp_units = [128]

        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
        return model
