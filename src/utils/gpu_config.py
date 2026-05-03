"""
GPU / CUDA configuration for TensorFlow + Keras.

Import this module **once**, as early as possible in any entry point that
builds or runs Keras models.  It is safe to import multiple times — the
configuration is applied only on the first call.

What this module does
---------------------
1. Enables **memory growth** for every detected GPU.
   Without this, TensorFlow pre-allocates all available VRAM at startup,
   which prevents other processes from sharing the GPU and can trigger OOM
   errors on small models.

2. Logs the detected GPU(s) so the user can confirm the correct device is
   being used.

3. Enables **mixed precision** (float16 compute, float32 storage) when a
   GPU with compute capability >= 7.0 is present.  On the RTX 3080
   (compute 8.6) this nearly doubles throughput for matrix multiplications
   while having negligible impact on accuracy for time-series forecasting.

Usage
-----
::

    # At the top of tfm_forecast.py, base_keras.py, or any entry point:
    from src.utils.gpu_config import configure_gpu
    configure_gpu()

Environment overrides
---------------------
* ``TF_GPU_MEMORY_LIMIT_MB`` (int) — if set, caps VRAM usage to that many
  MB via a virtual device instead of using growth mode.
* ``TF_DISABLE_MIXED_PRECISION=1`` — disables mixed precision even when
  a capable GPU is present (useful for debugging NaN gradients).

References
----------
* https://www.tensorflow.org/guide/gpu
* https://www.tensorflow.org/guide/mixed_precision
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_configured = False


def configure_gpu(mixed_precision: bool | None = None) -> None:
    """Configure TensorFlow GPU settings.

    Safe to call multiple times — configuration is applied only once.

    Parameters
    ----------
    mixed_precision : bool or None
        When ``True``, enable float16 mixed precision.
        When ``False``, disable it explicitly.
        When ``None`` (default), enable automatically on GPUs with
        compute capability >= 7.0 (Volta and newer), unless the
        ``TF_DISABLE_MIXED_PRECISION`` environment variable is set to ``1``.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # TensorFlow must be imported here (not at module level) so that this
    # module can be imported without triggering TF initialisation prematurely.
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning(
            "gpu_config: No GPU detected — training will run on CPU only. "
            "If a GPU is expected, check CUDA drivers and TF build."
        )
        return

    # ── 1. Memory growth ──────────────────────────────────────────────────
    memory_limit_mb = os.environ.get('TF_GPU_MEMORY_LIMIT_MB')
    if memory_limit_mb:
        try:
            limit = int(memory_limit_mb)
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit)],
                )
            logger.info(
                "gpu_config: GPU VRAM capped to %d MB (TF_GPU_MEMORY_LIMIT_MB).",
                limit,
            )
        except Exception as exc:
            logger.warning("gpu_config: Could not set memory limit: %s", exc)
    else:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                # Can happen if the GPU was already initialised
                logger.warning(
                    "gpu_config: set_memory_growth for %s failed: %s", gpu.name, exc
                )
        logger.info("gpu_config: Memory growth enabled for %d GPU(s).", len(gpus))

    # ── 2. Log device info ─────────────────────────────────────────────────
    # Trigger device initialisation so we get the full device name in logs
    logical_gpus = tf.config.list_logical_devices('GPU')
    logger.info(
        "gpu_config: %d physical GPU(s), %d logical GPU(s).",
        len(gpus), len(logical_gpus),
    )
    for gpu in gpus:
        logger.info("  GPU: %s", gpu.name)

    # ── 3. Mixed precision (optional) ─────────────────────────────────────
    _configure_mixed_precision(tf, gpus, mixed_precision)


def _configure_mixed_precision(tf, gpus, mixed_precision: bool | None) -> None:
    """Enable float16 mixed precision on capable GPUs."""
    if os.environ.get('TF_DISABLE_MIXED_PRECISION', '0') == '1':
        logger.info("gpu_config: Mixed precision disabled via TF_DISABLE_MIXED_PRECISION.")
        return

    if mixed_precision is False:
        logger.info("gpu_config: Mixed precision disabled by caller.")
        return

    if mixed_precision is True:
        _set_mixed_precision(tf)
        return

    # Auto-detect: enable only for compute capability >= 7.0 (Volta+)
    # TF does not expose compute capability directly; we use a GPU details
    # probe via the C API if available, otherwise default to enabling it
    # since modern CUDA-capable GPUs (RTX series) all qualify.
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        cc = details.get('compute_capability')
        if cc is not None:
            major = cc[0] if isinstance(cc, (list, tuple)) else int(str(cc)[0])
            if major >= 7:
                logger.info(
                    "gpu_config: Compute capability %s >= 7 — enabling mixed precision.",
                    cc,
                )
                _set_mixed_precision(tf)
            else:
                logger.info(
                    "gpu_config: Compute capability %s < 7 — mixed precision skipped.",
                    cc,
                )
            return
    except Exception:
        pass

    # Fallback: enable mixed precision (safe for any modern CUDA GPU)
    logger.info(
        "gpu_config: Could not determine compute capability — "
        "enabling mixed precision as default for CUDA GPU."
    )
    _set_mixed_precision(tf)


def _set_mixed_precision(tf) -> None:
    """Apply float16 global policy."""
    try:
        from keras import mixed_precision as mp
        mp.set_global_policy('mixed_float16')
        logger.info("gpu_config: Mixed precision policy set to 'mixed_float16'.")
    except Exception as exc:
        logger.warning("gpu_config: Could not set mixed precision: %s", exc)
