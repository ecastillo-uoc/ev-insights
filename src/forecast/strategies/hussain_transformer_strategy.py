"""
Simplified Transformer strategy reproducing the architecture from Hussain et al. (2025).

    Hussain, A., Eswarakrishnan, V., Aslam, A. & Tripura, S.
    "Charging stations demand forecasting using LSTM based hybrid transformer model."
    Sci Rep 15, 13555 (2025). https://doi.org/10.1038/s41598-025-20421-y

Architecture (from paper Fig. 3 and Eqs. 8–12):
  Input → Dense(encoding_dim, ReLU)       [Eq. 8]
        → MultiHeadAttention(dropout=0.2) [Eq. 9]
        → GlobalAveragePooling1D          [Eq. 10]
        → Dense(1)                        [Eq. 12]

Key differences from our standard ``TransformerModelStrategy``:
  * Single attention layer (no stacking)
  * No residual connections
  * No LayerNormalization
  * No feed-forward sub-block inside the encoder
  * No MLP head (direct Dense(1) after pooling)
  * Dropout 0.2 (vs. 0.1 in our variant)

Article hyperparameters (Table 1):
  epochs=100, batch_size=32, learning_rate=0.001,
  dropout=0.2, activation=ReLU, scaler=MinMaxScaler.
  look_back = prediction_period (30, 120, or 240 days).

All shared train/predict pipeline logic lives in ``KerasTimeSeriesBaseStrategy``.
"""

from typing import Tuple

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Dense, MultiHeadAttention, GlobalAveragePooling1D,
)

from .base_keras import KerasTimeSeriesBaseStrategy


class HussainTransformerModelStrategy(KerasTimeSeriesBaseStrategy):
    """Simplified single-attention Transformer from Hussain et al. (2025)."""

    @property
    def _params_key(self) -> str:
        return 'hussain_transformer_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'Hussain Transformer'

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        encoding_dim: int = 64,
        num_heads: int = 4,
        key_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        **_kwargs,
    ):
        """Build the simplified Transformer described in Hussain et al.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, 1)`` for univariate data.
        encoding_dim : int
            Output dimensionality of the initial Dense(ReLU) encoding layer.
        num_heads : int
            Number of parallel attention heads.
        key_dim : int
            Dimensionality of query/key projections in each attention head.
        dropout : float
            Dropout rate applied inside MultiHeadAttention.
        learning_rate : float
            Adam optimiser learning rate.
        """
        inputs = Input(shape=input_shape)

        # Dense encoding layer (Eq. 8 in paper)
        x = Dense(encoding_dim, activation='relu')(inputs)

        # Single MultiHeadAttention — no residual, no LayerNorm (Eq. 9)
        x = MultiHeadAttention(
            key_dim=key_dim, num_heads=num_heads, dropout=dropout,
        )(x, x)

        # Global average pooling (Eq. 10)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)

        # Regression output — single Dense(1), no MLP head (Eq. 12)
        outputs = Dense(1)(x)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model
