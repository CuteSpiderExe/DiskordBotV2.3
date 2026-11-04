"""Convert model audio to Discord PCM without loading another CUDA runtime."""

import math

import numpy as np
from scipy.signal import resample_poly


def stereo48k(mono: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if not np.isfinite(mono).all():
        raise ValueError("Non-finite TTS audio")
    if sample_rate != 48000:
        divisor = math.gcd(sample_rate, 48000)
        mono = resample_poly(mono, 48000 // divisor, sample_rate // divisor)
    fade = min(len(mono), 1440)
    if fade:
        mono = mono.copy()
        mono[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    pcm = (mono * 32767).clip(-32768, 32767).astype("<i2")
    return np.repeat(pcm[:, None], 2, axis=1).tobytes()
