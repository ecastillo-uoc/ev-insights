"""
Transformer encoder strategy for time-series forecasting.

Architecture (Functional API)::

  Input → Dense(d_model) → SinusoidalPE(look_back, d_model) →
          [N × TransformerEncoder blocks] → GlobalAveragePooling1D →
          MLP head (Dense(128, relu) + Dropout) → Dense(output_steps)

Each TransformerEncoder block::

  LayerNorm → MultiHeadAttention → residual Add →
  LayerNorm → Dense(ff_dim, relu) → Dropout → Dense(d_model) → residual Add

Hyperparameters read from ``df.attrs['transformer_params']``:
  epochs, batch_size, learning_rate, look_back,
  d_model, head_size, num_heads, ff_dim, num_transformer_blocks,
  mlp_units, dropout, mlp_dropout.

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, Dropout, Lambda,
    MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D,
)

from .base_keras import KerasTimeSeriesBaseStrategy, _sinusoidal_pe


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
        d_model: int = 64,
        head_size: int = 16,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_transformer_blocks: int = 2,
        mlp_units: list | None = None,
        dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        learning_rate: float = 0.001,
        output_steps: int = 1,
        **_kwargs,
    ):
        """Construct a multi-block Transformer encoder model.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, n_features)`` — typically ``(look_back, 1)``.
        d_model : int
            Internal embedding dimension.  Input features are projected
            to this space before the encoder blocks so that positional
            encoding and attention operate on a rich representation.
        head_size : int
            Key/query dimensionality per attention head (``key_dim`` in
            Keras).  Convention: ``head_size = d_model // num_heads``.
        num_heads : int
            Number of parallel attention heads.
        ff_dim : int
            Hidden units in the point-wise FFN inside each encoder block.
            Vaswani et al. (2017) use ``4 × d_model``; default 256 = 4 × 64.
        num_transformer_blocks : int
            Number of stacked encoder blocks.
        mlp_units : list[int]
            Widths of Dense layers in the MLP regression head.
        dropout, mlp_dropout : float
            Dropout rates for encoder blocks and MLP head respectively.
        learning_rate : float
            Adam optimiser learning rate.
        output_steps : int
            Number of output values.  1 for single-step; ``forecast_horizon``
            for direct multi-step models.
        """
        if mlp_units is None:
            mlp_units = [128]

        seq_len = input_shape[0]
        pe_matrix = _sinusoidal_pe(seq_len, d_model)  # (look_back, d_model)

        inputs = Input(shape=input_shape)
        # Project input features → d_model so attention has a rich embedding
        x = Dense(d_model)(inputs)
        # Sinusoidal positional encoding parameterised by look_back
        x = Lambda(
            lambda z, _pe=pe_matrix: z + _pe, name='positional_encoding',
        )(x)
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        # Pool over time dimension (look_back) → (batch, d_model)
        x = GlobalAveragePooling1D()(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(output_steps)(x)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])
        return model
