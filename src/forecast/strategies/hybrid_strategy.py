"""
Hybrid LSTM + Universal Transformer encoder for univariate time-series forecasting.

Architecture (Functional API)::

  Input(look_back, 1)
    → Dense(d_model)                     [input projection: 1 → d_model]
    + SinusoidalPE(look_back, d_model)
    → LSTM(d_model, return_sequences)    [recurrent local-pattern encoder]
    → [N shared Transformer blocks — Universal Transformer, Dehghani et al. 2019]
         shared MHA(key_dim=d_model//num_heads, num_heads, dropout)(x, x)
         Add([attn, x]) → shared LayerNorm
         shared Dense(ff_dim, relu) → Dropout → shared Dense(d_model)
         Add([ff, x]) → shared LayerNorm
    → GlobalAveragePooling1D             [pool over time dimension]
    → Dense(ff_dim // 2, relu)           [MLP regression head]
    → Dense(output_steps)

Design rationale
----------------
* **LSTM as local encoder** — the recurrent inductive bias captures short-range
  temporal patterns that pure attention misses on short sequences
  (Tay et al. 2020 "Long Range Arena").
* **Universal Transformer weight tying** — Dehghani et al. (ICLR 2019) show that
  N passes through a *single* tied Transformer block outperform N independent
  blocks on small-to-medium datasets by acting as a regulariser and halving the
  attention + FFN parameter count.  With ~1 000 daily training samples this is
  critical to avoid overfitting.
* **No decoder** — in backtest (rolling one-step-ahead) mode there is no target
  sequence to decode; the previous decoder LSTM was fed ``enc_out`` and then
  cross-attended back to ``enc_out``, adding zero new information.  A GAP + MLP
  head is the standard pooling mechanism for sequence regression (Devlin et al.
  2019 "BERT").
* **Input projection** — lifting the 1-D raw input to d_model before the LSTM
  allows the recurrent layer to operate in the same latent space as the
  attention queries/keys (Chorowski et al. 2015).

Hyperparameters read from ``df.attrs['hybrid_params']``:
  epochs, batch_size, learning_rate, look_back,
  d_model, num_heads, ff_dim, num_transformer_blocks, dropout.

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
    """LSTM encoder + Universal Transformer for univariate time-series forecasting.

    The LSTM captures local temporal patterns via recurrent inductive bias;
    N weight-tied Transformer blocks add global context with a fraction of the
    parameters of untied stacks (Dehghani et al., ICLR 2019).
    """

    @property
    def _params_key(self) -> str:
        return 'hybrid_params'

    @property
    def _strategy_display_name(self) -> str:
        return 'Hybrid LSTM-Transformer'

    def build_model(
        self,
        input_shape: Tuple[int, ...],
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        output_steps: int = 1,
        **_kwargs,
    ):
        """Build the LSTM + Universal-Transformer hybrid model.

        Parameters
        ----------
        input_shape : tuple
            ``(look_back, n_features)`` — univariate: ``(look_back, 1)``.
        d_model : int
            Hidden dimension for the input projection, LSTM units, and
            attention key dimension.  Default 64 (same as our Transformer),
            making parameter counts comparable.
        num_heads : int
            Number of parallel attention heads.
            ``key_dim = d_model // num_heads`` per head.
        ff_dim : int
            Hidden units in the shared FFN sub-block.
            Default 128 = 2 × d_model (Vaswani et al. 2017, §3.3).
        num_transformer_blocks : int
            Number of times the *shared* Transformer block is applied
            (Universal Transformer depth).  Default 2.
        dropout : float
            Dropout rate in attention and between FFN Dense layers.
        learning_rate : float
            Adam optimiser learning rate with ``clipnorm=1.0``.
        output_steps : int
            Output size.  1 for single-step backtest; ``forecast_horizon``
            for direct multi-step mode.
        """
        seq_len = input_shape[0]
        pe_matrix = _sinusoidal_pe(seq_len, d_model)  # (look_back, d_model)
        key_dim = max(1, d_model // num_heads)

        inputs = Input(shape=input_shape)

        # ── 1. Input projection + sinusoidal PE ───────────────────────
        # Lifts 1-D input into d_model space before the LSTM so the
        # recurrent layer operates in the same latent space as attention
        # queries/keys (Chorowski et al. 2015).
        x = Dense(d_model, name='input_proj')(inputs)          # (batch, look_back, d_model)
        x = Lambda(
            lambda z, _pe=pe_matrix: z + _pe, name='pe',
        )(x)

        # ── 2. LSTM encoder — local temporal pattern capture ──────────
        # return_sequences=True keeps the full (batch, look_back, d_model)
        # tensor for subsequent attention over all time steps.
        x = KerasLSTM(units=d_model, return_sequences=True, name='lstm_enc')(x)

        # ── 3. Universal Transformer — N passes through shared weights ─
        # Weight tying: the *same* MHA, FFN Dense, and LayerNorm objects
        # are called once per block, sharing their parameters across depth.
        # Reference: Dehghani et al. "Universal Transformers", ICLR 2019.
        shared_mha = MultiHeadAttention(
            key_dim=key_dim, num_heads=num_heads, dropout=dropout,
            name='shared_mha',
        )
        shared_ln_attn = LayerNormalization(epsilon=1e-6, name='shared_ln_attn')
        shared_ffn_up   = Dense(ff_dim, activation='relu', name='shared_ffn_up')
        shared_ffn_drop = Dropout(dropout, name='shared_ffn_drop')
        shared_ffn_down = Dense(d_model, name='shared_ffn_down')
        shared_ln_ffn   = LayerNormalization(epsilon=1e-6, name='shared_ln_ffn')

        for _ in range(num_transformer_blocks):
            # Self-attention sub-layer with residual connection
            attn = shared_mha(x, x)
            x = shared_ln_attn(Add()([attn, x]))
            # Position-wise FFN sub-layer with residual connection
            ff = shared_ffn_drop(shared_ffn_up(x))
            ff = shared_ffn_down(ff)
            x = shared_ln_ffn(Add()([ff, x]))

        # ── 4. Regression head ────────────────────────────────────────
        # GlobalAveragePooling collapses the time dimension (BERT-style
        # pooling; Devlin et al. 2019) before the final MLP.
        x = GlobalAveragePooling1D(name='gap')(x)              # (batch, d_model)
        x = Dense(ff_dim // 2, activation='relu', name='head_dense')(x)
        outputs = Dense(output_steps, name='output')(x)

        model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model
