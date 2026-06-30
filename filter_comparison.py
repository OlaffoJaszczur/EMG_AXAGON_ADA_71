"""
filter_comparison.py
────────────────────────────────────────────────────────────────────────────
EMG Powerline Noise Removal — FIR vs IIR x Notch vs Comb

Loads emg_line_L.csv from a recordings subfolder, applies a 6-500 Hz
bandpass, then demonstrates all four filter types:
  1. fir_notch_filter  — FIR bandstop cascade (firwin, Hamming window)
  2. iir_notch_filter  — IIR biquad notch cascade (iirnotch, Q=30)
  3. fir_comb_filter   — Feedforward FIR comb  y[n] = x[n] - x[n-M]
  4. iir_comb_filter   — Feedback IIR comb     y[n] = x[n] - x[n-M] + r^M*y[n-M]

Run:
  python filter_comparison.py
  python filter_comparison.py emg_rec_20260616_113306
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sig

# ── recording settings ────────────────────────────────────────────────────────
RECORDINGS_DIR  = Path("recordings")
DEFAULT_FOLDER  = "emg_rec_20260625_092258"
CHANNEL         = "emg_line_L"

# ── filter / signal parameters ────────────────────────────────────────────────
F0           = 49.97328   # Hz — actual mains frequency (measured, not exactly 50)
HARMONICS_MAX = 500.0      # Hz — remove all harmonics of F0 up to this frequency

# ── FIR notch tap cap ─────────────────────────────────────────────────────────
# Ideal N = 8*fs/bw. At fs=44100, bw=5 Hz this is ~70 000 taps — impractical.
# We cap at MAX_FIR_TAPS so the demo runs in reasonable time. The resulting
# notch is much wider than ideal; this is intentional: it shows why FIR notch
# is unsuitable at audio sampling rates.
MAX_FIR_TAPS = 4001


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_recording(folder: str = DEFAULT_FOLDER,
                   channel: str = CHANNEL,
                   recordings_dir: Path = RECORDINGS_DIR):
    """
    Load one channel CSV from a recording subfolder.

    CSV format expected: columns 'time' (s) and 'data' (V).
    fs is derived from the sample interval (no assumption of exact value).

    Returns (t, x, fs).
    """
    path = recordings_dir / folder / f"{channel}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    df = pd.read_csv(path)
    t  = df["time"].values.astype(float)
    x  = df["data"].values.astype(float)
    fs = 1.0 / (t[1] - t[0])   # derive from actual sample interval

    print(f"Loaded  : {path}")
    print(f"Samples : {len(x):,}   fs = {fs:.4f} Hz   duration = {t[-1]:.3f} s")
    return t, x, fs


# ══════════════════════════════════════════════════════════════════════════════
# BANDPASS PRE-FILTER
# ══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(x: np.ndarray, fs: float,
                    low_hz: float = 6.0, high_hz: float = 500.0,
                    order: int = 4) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth bandpass 6-500 Hz applied via sosfiltfilt.

    This is always the first stage before any notch/comb filter. It:
      - Removes DC and very-low-frequency electrode drift (below 6 Hz).
      - Limits the signal to the physiologically relevant EMG band (up to 500 Hz).
      - Prevents aliasing artefacts from the comb filters that also null DC.
    """
    sos = sig.butter(order, [low_hz, high_hz], btype='band', fs=fs, output='sos')
    return sig.sosfiltfilt(sos, x)


# ══════════════════════════════════════════════════════════════════════════════
# FILTER 1 — FIR NOTCH
# ══════════════════════════════════════════════════════════════════════════════

def fir_notch_filter(x: np.ndarray, fs: float,
                     f0: float = F0, bw: float = 5.0,
                     max_hz: float = HARMONICS_MAX,
                     max_taps: int = MAX_FIR_TAPS) -> np.ndarray:
    """
    Cascade of FIR bandstop (notch) filters, one per harmonic of f0.
    Design: scipy.signal.firwin with Hamming window.
    Applied zero-phase via filtfilt (offline only).

    FIR = Finite Impulse Response = only ZEROS in the transfer function,
    no feedback, no poles. The filter is a weighted sum of past inputs only.

    Transfer function of one stage:
        H(z) = b0 + b1*z^{-1} + ... + b_{N-1}*z^{-(N-1)}
    All N tap weights (b coefficients) are symmetric -> linear phase.

    Key EMG trade-offs
    ------------------
    + Perfectly linear phase: every frequency is delayed by exactly (N-1)/2
      samples, preserving the EMG waveform shape without distortion.
    + Unconditionally stable - no recursive computation, no poles.
    - Needs many taps for a sharp notch: N ~= 8*fs / bw (Hamming window).
      At fs = 2000 Hz, bw = 5 Hz -> N ~= 3201 taps per harmonic.
      At fs = 44100 Hz the same notch needs ~70 000 taps — impractical!
      This is the fundamental limitation of FIR for power-line rejection.
    - max_taps cap (default 4001 at 44100 Hz) means transition bands are
      ~88 Hz wide instead of 5 Hz: the notch in the FFT will appear VERY
      WIDE, demonstrating why FIR notch is ill-suited to audio-rate EMG.
    """
    n_taps = int(8 * fs / bw)
    if n_taps % 2 == 0:
        n_taps += 1  # firwin bandstop requires an odd number of taps

    capped = n_taps > max_taps
    if capped:
        n_taps = max_taps if max_taps % 2 != 0 else max_taps - 1
        actual_bw = 8 * fs / n_taps
        print(f"  [FIR Notch] tap count capped at {n_taps} "
              f"(ideal {int(8*fs/bw)+1}, transition band ~{actual_bw:.0f} Hz "
              f"instead of {bw} Hz -- notch will be very wide at fs={fs:.0f} Hz)")

    y = x.copy().astype(float)
    k = 1
    while k * f0 <= max_hz and k * f0 < fs / 2:
        fc = k * f0
        b = sig.firwin(n_taps,
                       [max(fc - bw / 2, 0.5), min(fc + bw / 2, fs / 2 - 0.5)],
                       window='hamming', pass_zero=True, fs=fs)
        y = sig.filtfilt(b, [1.0], y)
        k += 1
    return y


# ══════════════════════════════════════════════════════════════════════════════
# FILTER 2 — IIR NOTCH
# ══════════════════════════════════════════════════════════════════════════════

def iir_notch_filter(x: np.ndarray, fs: float,
                     f0: float = F0, Q: float = 30.0,
                     max_hz: float = HARMONICS_MAX) -> np.ndarray:
    """
    Cascade of 2nd-order IIR biquad notch filters at each harmonic of f0.
    Applied zero-phase via filtfilt (forward + backward pass).

    IIR = Infinite Impulse Response = filter uses both ZEROS and POLES.
    The 2nd-order notch (iirnotch) places:
      * Two zeros exactly ON the unit circle at +/-f_c -> infinite attenuation.
      * Two poles just INSIDE the unit circle, close to the zeros
        -> sharp, narrow notch; passband is virtually untouched.

    Q controls sharpness:  Q = f0 / notch_bandwidth.
    Q = 30  ->  ~1.7 Hz notch width at 50 Hz  (standard for clinical EMG).

    Key EMG trade-offs
    ------------------
    + Extremely sharp notch with only 2 poles + 2 zeros per harmonic.
    + Same computation cost at any fs — ideal for audio-rate EMG (44100 Hz).
    + Flattest possible passband between harmonics -> minimum EMG distortion.
    - Not linear phase: different frequencies see different group delays.
      filtfilt (double-pass) achieves zero-phase but requires the full
      signal in memory — unusable for real-time streaming.
    - Very high Q (> 100) can cause numerical issues in single precision.
    """
    y = x.copy().astype(float)
    k = 1
    while k * f0 <= max_hz and k * f0 < fs / 2:
        b, a = sig.iirnotch(k * f0, Q, fs=fs)
        y = sig.filtfilt(b, a, y)
        k += 1
    return y


# ══════════════════════════════════════════════════════════════════════════════
# FILTER 3 — FIR COMB (feedforward)
# ══════════════════════════════════════════════════════════════════════════════

def fir_comb_filter(x: np.ndarray, fs: float,
                    f0: float = F0) -> np.ndarray:
    """
    Feedforward FIR comb filter:  y[n] = x[n] - x[n-M],  M = round(fs / f0).
    Applied zero-phase via filtfilt.

    Transfer function:  H(z) = 1 - z^{-M}
    This places M zeros EQUALLY SPACED on the unit circle:
        z_k = e^{j*2*pi*k/M},  k = 0, 1, ..., M-1
    -> simultaneous nulls at DC, f0, 2*f0, 3*f0, ..., (M-1)*f0.

    For fs = 44100 Hz and f0 = 49.97328 Hz:  M = 882 samples.
    For fs = 2000  Hz and f0 = 50 Hz:        M = 40 samples.

    Key EMG trade-offs
    ------------------
    + Removes ALL harmonics of f0 in a single O(1) pass — ultra-efficient.
    + Linear phase (symmetric FIR of length M+1) -> waveform shape preserved.
    + Trivially simple: one delay line and one subtraction.
    - Passband is NOT flat: magnitude follows 2|sin(pi*f*M/fs)|, peaking at
      2x gain midway between harmonics. filtfilt squares this to 4x (+12 dB)
      at those mid-frequencies. This DISTORTS the EMG significantly.
    - Also notches DC — harmless after the 6-500 Hz bandpass.
    """
    M = int(round(fs / f0))
    b = np.zeros(M + 1)
    b[0] = 1.0
    b[M] = -1.0
    return sig.filtfilt(b, [1.0], x.astype(float))


# ══════════════════════════════════════════════════════════════════════════════
# FILTER 4 — IIR COMB (feedback)
# ══════════════════════════════════════════════════════════════════════════════

def iir_comb_filter(x: np.ndarray, fs: float,
                    f0: float = F0, r_m: float = 0.9) -> np.ndarray:
    """
    Feedback IIR comb filter.
    Difference equation:  y[n] = x[n] - x[n-M] + r_m * y[n-M]
    Transfer function:    H(z) = (1 - z^{-M}) / (1 - r_m * z^{-M})

    Zeros at z_k = e^{j*2*pi*k/M}           (on unit circle — same as FIR comb)
    Poles at z_k = r_m^{1/M} * e^{j*2*pi*k/M}   (inside unit circle, near zeros)

    r_m is the DIRECT feedback coefficient (= r^M in classical notation).
    Parameterising by r_m makes the notch bandwidth fs-INDEPENDENT:
        BW_3dB ~= (1 - r_m) * f0 / pi   [Hz]

    Examples at f0 = 49.97 Hz:
        r_m = 0.90  ->  BW ~= 1.6 Hz  (default — clearly sharper than FIR comb)
        r_m = 0.98  ->  BW ~= 0.3 Hz  (very sharp, clinical quality)
        r_m = 0.50  ->  BW ~= 7.9 Hz  (wide, safe for real-time)

    IMPORTANT: if you parameterise as r^M (the classical approach), r must be
    close to 1 at high fs.  At fs = 44100, f0 = 50 Hz, M = 882:
        r = 0.98  ->  r^M = 0.98^882 ~= 2e-8  (nearly 0 — same as FIR comb!)
    Using r_m directly avoids this silent failure.

    Key EMG trade-offs
    ------------------
    + Sharpest notches of all four filters for the lowest computational cost.
    + Flat passband between harmonics -> best EMG signal preservation.
    + Removes all harmonics in one pass, like FIR comb.
    + r_m parameter is fs-independent — same value works at any sample rate.
    - Not linear phase; filtfilt achieves zero-phase for offline use.
    - Stability requires r_m < 1; values above 0.99 may ring at high fs.
    - Also nulls DC (harmless after the 6-500 Hz bandpass).
    """
    M = int(round(fs / f0))

    b = np.zeros(M + 1)
    a = np.zeros(M + 1)
    b[0] = 1.0;  b[M] = -1.0   # numerator:   1 - z^{-M}
    a[0] = 1.0;  a[M] = -r_m   # denominator: 1 - r_m * z^{-M}

    bw_approx = (1 - r_m) * f0 / np.pi
    print(f"  [IIR Comb] M={M}, r_m={r_m}, approx notch BW={bw_approx:.2f} Hz per harmonic")

    return sig.filtfilt(b, a, x.astype(float))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_fft(x: np.ndarray, fs: float):
    N = len(x)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag   = np.abs(np.fft.rfft(x)) * (2.0 / N)
    mag[0] /= 2
    if N % 2 == 0:
        mag[-1] /= 2
    return freqs, mag


def single_stage_responses(fs: float, f0: float = F0,
                            bw: float = 5.0, Q: float = 30.0,
                            r_m: float = 0.9, nfft: int = 16384):
    """
    Compute single-stage magnitude (dB) for each filter type.
    Used for the frequency-response panels in the plot.
    """
    M = int(round(fs / f0))

    # FIR notch — single stage, tap count capped as in the actual filter
    n_taps = int(8 * fs / bw)
    if n_taps % 2 == 0:
        n_taps += 1
    if n_taps > MAX_FIR_TAPS:
        n_taps = MAX_FIR_TAPS if MAX_FIR_TAPS % 2 != 0 else MAX_FIR_TAPS - 1
    b_fn = sig.firwin(n_taps,
                      [max(f0 - bw / 2, 0.5), min(f0 + bw / 2, fs / 2 - 0.5)],
                      window='hamming', pass_zero=True, fs=fs)
    w_fn, H_fn = sig.freqz(b_fn, [1.0], worN=nfft, fs=fs)

    # IIR notch — single biquad at f0
    b_in, a_in = sig.iirnotch(f0, Q, fs=fs)
    w_in, H_in = sig.freqz(b_in, a_in, worN=nfft, fs=fs)

    # FIR comb
    b_fc = np.zeros(M + 1);  b_fc[0] = 1.0;  b_fc[M] = -1.0
    w_fc, H_fc = sig.freqz(b_fc, [1.0], worN=nfft, fs=fs)

    # IIR comb
    b_ic = np.zeros(M + 1);  a_ic = np.zeros(M + 1)
    b_ic[0] = 1.0;  b_ic[M] = -1.0
    a_ic[0] = 1.0;  a_ic[M] = -r_m
    w_ic, H_ic = sig.freqz(b_ic, a_ic, worN=nfft, fs=fs)

    def dB(H):
        return 20 * np.log10(np.abs(H) + 1e-14)

    return {
        'FIR Notch': (w_fn, dB(H_fn)),
        'IIR Notch': (w_in, dB(H_in)),
        'FIR Comb':  (w_fc, dB(H_fc)),
        'IIR Comb':  (w_ic, dB(H_ic)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'raw':        '#AAAAAA',
    'bandpassed': '#1565C0',
    'FIR Notch':  '#E65100',
    'IIR Notch':  '#6A1B9A',
    'FIR Comb':   '#2E7D32',
    'IIR Comb':   '#C62828',
}

LABELS = {
    'FIR Notch': 'FIR Notch  (firwin, N<=4001, zero-phase)',
    'IIR Notch': 'IIR Notch  (iirnotch, Q=30, zero-phase)',
    'FIR Comb':  'FIR Comb   y[n]=x[n]-x[n-M]  (zero-phase)',
    'IIR Comb':  'IIR Comb   +r^M*y[n-M], r=0.98  (zero-phase)',
}


def plot_demo(t, x_raw, x_bp, filtered, ffts, responses,
              fs: float = 44100.0, f0: float = F0, folder: str = ""):
    """
    5-row figure:
      Row 0: freq response — notch zoom (left) | comb full spectrum (right)
      Row 1: freq response — all four compared, zoomed at f0
      Row 2: time domain — 0.5 s excerpt
      Row 3: FFT full spectrum (log, 0-530 Hz)
      Row 4: FFT zoom at f0 (left) | FFT zoom at 2*f0 (right)
    """
    fft_lim = min(530.0, fs / 2)
    f1_lo, f1_hi = f0 - 8, f0 + 8
    f2_lo, f2_hi = 2 * f0 - 8, 2 * f0 + 8

    fig = plt.figure(figsize=(18, 26), constrained_layout=True)
    fig.suptitle(
        f"EMG Powerline Noise Removal — FIR vs IIR x Notch vs Comb\n"
        f"Recording: {folder}/{CHANNEL}.csv  |  "
        f"fs = {fs:.1f} Hz  |  f0 = {f0:.5f} Hz  |  "
        f"BP 6-500 Hz then 4 filters (all zero-phase via filtfilt)",
        fontsize=12, fontweight='bold',
    )
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           height_ratios=[1.2, 1.1, 1.0, 1.1, 1.1])

    # ── (0,0): notch single-stage zoom ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(
        f"Freq response — NOTCH (single stage, zoom +-15 Hz around f0)\n"
        f"FIR capped at {MAX_FIR_TAPS} taps -> transition band ~{8*fs/MAX_FIR_TAPS:.0f} Hz wide (vs 5 Hz ideal)",
        fontsize=9,
    )
    for name in ('FIR Notch', 'IIR Notch'):
        w, H = responses[name]
        ax.plot(w, H, color=COLORS[name], lw=2.0, label=name)
    ax.axvline(f0, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlim(f0 - 15, f0 + 15);  ax.set_ylim(-80, 3)
    ax.set_xlabel("Frequency (Hz)");  ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

    # ── (0,1): comb full spectrum ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(
        "Freq response — COMB (single stage, 0 to 530 Hz)\n"
        "Both remove all harmonics at once; IIR has much narrower notches",
        fontsize=9,
    )
    for name in ('FIR Comb', 'IIR Comb'):
        w, H = responses[name]
        ax.plot(w, H, color=COLORS[name], lw=1.8, label=name)
    for k in range(1, int(fft_lim / f0) + 1):
        ax.axvline(k * f0, color='grey', ls=':', lw=0.5, alpha=0.35)
    ax.set_xlim(0, fft_lim);  ax.set_ylim(-80, 3)
    ax.set_xlabel("Frequency (Hz)");  ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

    # ── (1,:): all four around f0 ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    ax.set_title(
        f"All four compared — zoom {f0-15:.1f} to {f0+15:.1f} Hz  "
        "(IIR variants have flat passband; FIR Comb is widest; FIR Notch is very wide at this fs)",
        fontsize=9,
    )
    for name in ('FIR Notch', 'IIR Notch', 'FIR Comb', 'IIR Comb'):
        w, H = responses[name]
        ax.plot(w, H, color=COLORS[name], lw=2.0, label=name)
    ax.axvline(f0, color='red', ls='--', lw=0.9, alpha=0.5,
               label=f'f0 = {f0:.5f} Hz')
    ax.set_xlim(f0 - 15, f0 + 15);  ax.set_ylim(-80, 3)
    ax.set_xlabel("Frequency (Hz)");  ax.set_ylabel("Magnitude (dB)")
    ax.legend(fontsize=9, ncol=3);  ax.grid(True, alpha=0.3)

    # ── (2,:): time domain excerpt ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    t0_plot, t1_plot = 1.0, 1.5
    tm = (t >= t0_plot) & (t < t1_plot)
    ax.set_title(
        "Time domain — 0.5 s excerpt  "
        "(grey = raw, dashed blue = bandpassed 6-500 Hz before harmonic removal)",
        fontsize=9,
    )
    ax.plot(t[tm], x_raw[tm], color=COLORS['raw'],        lw=0.5, alpha=0.55,
            label='Raw')
    ax.plot(t[tm], x_bp[tm],  color=COLORS['bandpassed'], lw=1.2, ls='--',
            label='Bandpassed 6-500 Hz (input to filters)')
    for name, y in filtered.items():
        ax.plot(t[tm], y[tm], color=COLORS[name], lw=0.9, label=LABELS[name])
    ax.set_xlabel("Time (s)");  ax.set_ylabel("Amplitude (V)")
    ax.legend(ncol=2, fontsize=7.5);  ax.grid(True, alpha=0.3)

    # ── (3,:): full FFT ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, :])
    ax.set_title(
        f"FFT magnitude — 0 to {fft_lim:.0f} Hz (log scale)  "
        f"| red dotted = harmonics of f0={f0:.5f} Hz",
        fontsize=9,
    )
    fr_r, fft_r = ffts['raw']
    fr_b, fft_b = ffts['bandpassed']
    ax.semilogy(fr_r, fft_r, color=COLORS['raw'],        lw=0.5, alpha=0.4,
                label='Raw')
    ax.semilogy(fr_b, fft_b, color=COLORS['bandpassed'], lw=1.0, ls='--',
                label='Bandpassed')
    for name in ('FIR Notch', 'IIR Notch', 'FIR Comb', 'IIR Comb'):
        fr_f, fft_f = ffts[name]
        ax.semilogy(fr_f, fft_f, color=COLORS[name], lw=1.0, label=name)
    for k in range(1, int(HARMONICS_MAX / f0) + 2):
        ax.axvline(k * f0, color='red', ls=':', lw=0.5, alpha=0.22)
    ax.set_xlim(0, fft_lim)
    ax.set_xlabel("Frequency (Hz)");  ax.set_ylabel("Magnitude (log)")
    ax.legend(ncol=3, fontsize=8);  ax.grid(True, alpha=0.3, which='both')

    # ── (4,0) and (4,1): FFT zoom at f0 and 2*f0 ─────────────────────────────
    for col, (f_center, fl, fh) in enumerate([
        (f0,     f1_lo, f1_hi),
        (2 * f0, f2_lo, f2_hi),
    ]):
        ax = fig.add_subplot(gs[4, col])
        ax.set_title(
            f"FFT zoom: {f_center:.2f} Hz — notch depth & width", fontsize=9,
        )
        ax.semilogy(fr_r, fft_r, color=COLORS['raw'],        lw=1.3, alpha=0.5,
                    label='Raw')
        ax.semilogy(fr_b, fft_b, color=COLORS['bandpassed'], lw=1.2, ls='--',
                    label='Bandpassed')
        for name in ('FIR Notch', 'IIR Notch', 'FIR Comb', 'IIR Comb'):
            fr_f, fft_f = ffts[name]
            zm = (fr_f >= fl) & (fr_f <= fh)
            ax.semilogy(fr_f[zm], fft_f[zm], color=COLORS[name], lw=1.3,
                        label=name)
        ax.axvline(f_center, color='red', ls=':', lw=0.9, alpha=0.5)
        ax.set_xlim(fl, fh)
        ax.set_xlabel("Frequency (Hz)");  ax.set_ylabel("Magnitude (log)")
        ax.legend(fontsize=7.5);  ax.grid(True, alpha=0.3, which='both')

    fig.savefig("filter_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved -> filter_comparison.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — FULL SIGNAL OVERLAY (time + FFT in one figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_overlay(t, x_raw, x_bp, filtered, ffts,
                 fs: float = 44100.0, f0: float = F0, folder: str = ""):
    """
    Single figure, two panels:
      Top   : full time-domain recording, all 6 cases overlaid.
      Bottom: FFT (0-530 Hz, log scale), all 6 cases overlaid.

    To keep rendering fast at 44100 Hz, the time traces are decimated to
    at most 20 000 display points (no information lost at figure scale).
    """
    fft_lim = min(530.0, fs / 2)
    step = max(1, len(t) // 20000)   # decimate for display only

    cases = [
        ('Raw',                 x_raw,               'raw'),
        ('Bandpassed 6-500 Hz', x_bp,                'bandpassed'),
        ('FIR Notch',           filtered['FIR Notch'], 'FIR Notch'),
        ('IIR Notch',           filtered['IIR Notch'], 'IIR Notch'),
        ('FIR Comb',            filtered['FIR Comb'],  'FIR Comb'),
        ('IIR Comb',            filtered['IIR Comb'],  'IIR Comb'),
    ]

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(18, 10),
                                      constrained_layout=True)
    fig.suptitle(
        f"Full signal — all cases overlaid  |  {folder}/{CHANNEL}\n"
        f"fs = {fs:.1f} Hz  |  f0 = {f0:.5f} Hz  |  duration = {t[-1]:.2f} s",
        fontsize=12, fontweight='bold',
    )

    # ── time domain ──────────────────────────────────────────────────────────
    ax_t.set_title("Full time domain — all cases overlaid", fontsize=10)
    for label, y, key in cases:
        lw    = 0.35 if key == 'raw' else 0.65
        alpha = 0.45 if key in ('raw', 'bandpassed') else 0.85
        ax_t.plot(t[::step], y[::step], color=COLORS[key],
                  lw=lw, alpha=alpha, label=label)
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude (V)")
    ax_t.legend(ncol=3, fontsize=9)
    ax_t.grid(True, alpha=0.3)

    # ── FFT ──────────────────────────────────────────────────────────────────
    ax_f.set_title(
        f"FFT — all cases overlaid  (0-{fft_lim:.0f} Hz, log scale)"
        f"  |  red dotted = harmonics of f0",
        fontsize=10,
    )
    for label, _, key in cases:
        fr, mag = ffts[key]
        zm = fr <= fft_lim
        lw    = 0.5 if key == 'raw' else 1.0
        alpha = 0.4 if key == 'raw' else 0.9
        ax_f.semilogy(fr[zm], mag[zm], color=COLORS[key],
                      lw=lw, alpha=alpha, label=label)
    for k in range(1, int(HARMONICS_MAX / f0) + 2):
        ax_f.axvline(k * f0, color='red', ls=':', lw=0.5, alpha=0.2)
    ax_f.set_xlim(0, fft_lim)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude (log)")
    ax_f.legend(ncol=3, fontsize=9)
    ax_f.grid(True, alpha=0.3, which='both')

    fig.savefig("filter_comparison_overlay.png", dpi=150, bbox_inches='tight')
    print("Saved -> filter_comparison_overlay.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — SEPARATE SUBPLOTS: TIME DOMAIN
# ══════════════════════════════════════════════════════════════════════════════

def plot_subplots_time(t, x_raw, x_bp, filtered,
                       fs: float = 44100.0, f0: float = F0, folder: str = ""):
    """
    3 rows x 2 columns of time-domain subplots, one per case.
    Layout:  Raw | Bandpassed
             FIR Notch | IIR Notch
             FIR Comb  | IIR Comb
    """
    step = max(1, len(t) // 20000)

    cases = [
        ('Raw',                 x_raw,               'raw'),
        ('Bandpassed 6-500 Hz', x_bp,                'bandpassed'),
        ('FIR Notch',           filtered['FIR Notch'], 'FIR Notch'),
        ('IIR Notch',           filtered['IIR Notch'], 'IIR Notch'),
        ('FIR Comb',            filtered['FIR Comb'],  'FIR Comb'),
        ('IIR Comb',            filtered['IIR Comb'],  'IIR Comb'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), constrained_layout=True)
    fig.suptitle(
        f"Full time domain — separate subplots  |  {folder}/{CHANNEL}\n"
        f"fs = {fs:.1f} Hz  |  duration = {t[-1]:.2f} s",
        fontsize=12, fontweight='bold',
    )

    for ax, (label, y, key) in zip(axes.flatten(), cases):
        rms = np.sqrt(np.mean(y ** 2))
        ax.plot(t[::step], y[::step], color=COLORS[key], lw=0.45)
        ax.set_title(label, fontsize=10, fontweight='bold', color=COLORS[key])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.grid(True, alpha=0.3)
        ax.text(0.99, 0.97, f"RMS = {rms:.5f} V",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='dimgrey')

    fig.savefig("filter_comparison_time_subplots.png", dpi=150,
                bbox_inches='tight')
    print("Saved -> filter_comparison_time_subplots.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — SEPARATE SUBPLOTS: FFT
# ══════════════════════════════════════════════════════════════════════════════

def plot_subplots_fft(ffts,
                      fs: float = 44100.0, f0: float = F0, folder: str = ""):
    """
    3 rows x 2 columns of FFT subplots, one per case.
    Layout:  Raw | Bandpassed
             FIR Notch | IIR Notch
             FIR Comb  | IIR Comb
    Each subplot has harmonic marker lines and shares the same x range.
    """
    fft_lim = min(530.0, fs / 2)

    case_order = [
        ('Raw',                 'raw'),
        ('Bandpassed 6-500 Hz', 'bandpassed'),
        ('FIR Notch',           'FIR Notch'),
        ('IIR Notch',           'IIR Notch'),
        ('FIR Comb',            'FIR Comb'),
        ('IIR Comb',            'IIR Comb'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), constrained_layout=True)
    fig.suptitle(
        f"FFT — separate subplots  (0-{fft_lim:.0f} Hz, log scale)"
        f"  |  {folder}/{CHANNEL}\n"
        f"Red dotted lines = harmonics of f0 = {f0:.5f} Hz",
        fontsize=12, fontweight='bold',
    )

    for ax, (label, key) in zip(axes.flatten(), case_order):
        fr, mag = ffts[key]
        zm = fr <= fft_lim
        ax.semilogy(fr[zm], mag[zm], color=COLORS[key], lw=0.8)
        for k in range(1, int(HARMONICS_MAX / f0) + 2):
            ax.axvline(k * f0, color='red', ls=':', lw=0.6, alpha=0.3)
        ax.set_title(label, fontsize=10, fontweight='bold', color=COLORS[key])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (log)")
        ax.set_xlim(0, fft_lim)
        ax.grid(True, alpha=0.3, which='both')
        # Annotate peak at f0
        mask_f0 = (fr >= f0 - 2) & (fr <= f0 + 2)
        if mask_f0.any():
            peak_v = float(np.max(mag[mask_f0]))
            ax.annotate(
                f"f0 peak\n{20*np.log10(peak_v+1e-20):.1f} dBV",
                xy=(f0, peak_v),
                xytext=(f0 + 15, peak_v * 3),
                fontsize=7, color='red', alpha=0.7,
                arrowprops=dict(arrowstyle='->', color='red', lw=0.7),
            )

    fig.savefig("filter_comparison_fft_subplots.png", dpi=150,
                bbox_inches='tight')
    print("Saved -> filter_comparison_fft_subplots.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — LINEAR-SCALE FFT SUBPLOTS (0-800 Hz)
# ══════════════════════════════════════════════════════════════════════════════

def plot_subplots_fft_linear(ffts,
                              fs: float = 44100.0, f0: float = F0,
                              folder: str = ""):
    """
    3 rows x 2 columns of FFT subplots — linear amplitude scale, 0-800 Hz.
    Identical layout to plot_subplots_fft but without the log compression so
    the absolute height of each harmonic spike is directly visible.
    """
    fft_lim = min(800.0, fs / 2)

    case_order = [
        ('Raw',                 'raw'),
        ('Bandpassed 6-500 Hz', 'bandpassed'),
        ('FIR Notch',           'FIR Notch'),
        ('IIR Notch',           'IIR Notch'),
        ('FIR Comb',            'FIR Comb'),
        ('IIR Comb',            'IIR Comb'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 15), constrained_layout=True)
    fig.suptitle(
        f"FFT — linear amplitude scale  (0-{fft_lim:.0f} Hz)"
        f"  |  {folder}/{CHANNEL}\n"
        f"Red dotted lines = harmonics of f0 = {f0:.5f} Hz",
        fontsize=12, fontweight='bold',
    )

    for ax, (label, key) in zip(axes.flatten(), case_order):
        fr, mag = ffts[key]
        zm = fr <= fft_lim
        ax.plot(fr[zm], mag[zm], color=COLORS[key], lw=0.8)
        for k in range(1, int(fft_lim / f0) + 2):
            ax.axvline(k * f0, color='red', ls=':', lw=0.6, alpha=0.3)
        ax.set_title(label, fontsize=10, fontweight='bold', color=COLORS[key])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (V/bin)")
        ax.set_xlim(0, fft_lim)
        ax.grid(True, alpha=0.3)
        # Annotate the f0 peak value
        mask_f0 = (fr >= f0 - 2) & (fr <= f0 + 2)
        if mask_f0.any():
            peak_v  = float(np.max(mag[mask_f0]))
            peak_fi = int(np.argmax(mag[mask_f0 & (fr <= fft_lim)]))
            ax.annotate(
                f"f0 = {peak_v:.4f}",
                xy=(f0, peak_v),
                xytext=(f0 + 30, peak_v * 0.85 + ax.get_ylim()[1] * 0.05),
                fontsize=7, color='red', alpha=0.8,
                arrowprops=dict(arrowstyle='->', color='red', lw=0.7),
            )

    fig.savefig("filter_comparison_fft_linear.png", dpi=150,
                bbox_inches='tight')
    print("Saved -> filter_comparison_fft_linear.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — PERFORMANCE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def plot_performance_table(x_bp, filtered, timings, ffts,
                            fs: float = 44100.0, f0: float = F0,
                            folder: str = ""):
    """
    Matplotlib figure displaying a color-coded performance table:
      Rows : Bandpassed (reference) + each of the 4 filters
      Cols : RMS (V), Processing time (s), attenuation (dB) at every harmonic
             of f0 up to HARMONICS_MAX.

    Cell color key for attenuation cells:
      Dark green  : > 20 dB   (excellent)
      Light green : 10-20 dB  (good)
      Yellow      : 3-10 dB   (marginal)
      Light red   : 0-3 dB    (poor)
      Red         : < 0 dB    (amplification)

    RMS cell color (ratio to bandpassed RMS):
      Green  : within +/- 10 %
      Yellow : within +/- 50 %
      Red    : > 50 % off
    """
    def _peak(key, f_target, f_bw=1.5):
        fr, mag = ffts[key]
        mask = (fr >= f_target - f_bw) & (fr <= f_target + f_bw)
        return float(np.max(mag[mask])) if mask.any() else 1e-20

    harmonics = []
    k = 1
    while k * f0 <= HARMONICS_MAX:
        harmonics.append(k * f0)
        k += 1
    n_h = len(harmonics)

    ref_peaks = [_peak('bandpassed', h) for h in harmonics]
    rms_bp    = float(np.sqrt(np.mean(x_bp ** 2)))

    # ── build row data ────────────────────────────────────────────────────────
    filter_names = list(filtered.keys())

    rows_text  = []   # list of lists of cell strings
    rows_color = []   # list of lists of cell colors

    def _atten_color(db):
        if db > 20:   return '#A5D6A7'   # dark green
        if db > 10:   return '#C8E6C9'   # light green
        if db > 3:    return '#FFF9C4'   # yellow
        if db >= 0:   return '#FFCDD2'   # light red
        return '#EF9A9A'                  # red (amplification)

    def _rms_color(rms):
        ratio = abs(rms / rms_bp - 1)
        if ratio <= 0.10: return '#C8E6C9'
        if ratio <= 0.50: return '#FFF9C4'
        return '#FFCDD2'

    # Bandpassed reference row
    ref_text  = ['Bandpassed (ref)', f'{rms_bp:.5f}', '---']
    ref_color = ['#E3F2FD', '#C8E6C9', '#E3F2FD']
    for _ in harmonics:
        ref_text.append('(ref)')
        ref_color.append('#E3F2FD')
    rows_text.append(ref_text)
    rows_color.append(ref_color)

    for name in filter_names:
        y    = filtered[name]
        rms  = float(np.sqrt(np.mean(y ** 2)))
        t_s  = timings.get(name, float('nan'))
        row_t = [name, f'{rms:.5f}', f'{t_s:.2f} s']
        row_c = ['#FAFAFA', _rms_color(rms), '#FAFAFA']
        for h, rp in zip(harmonics, ref_peaks):
            p     = _peak(name, h)
            atten = 20 * np.log10(rp / (p + 1e-20))
            row_t.append(f'{atten:+.1f} dB')
            row_c.append(_atten_color(atten))
        rows_text.append(row_t)
        rows_color.append(row_c)

    # ── build column labels ───────────────────────────────────────────────────
    harm_cols = [f'{k+1}xf0\n{h:.1f} Hz' for k, h in enumerate(harmonics)]
    col_labels = ['Filter', 'RMS (V)', 'Time'] + harm_cols

    # ── figure ────────────────────────────────────────────────────────────────
    n_rows = len(rows_text)
    n_cols = len(col_labels)

    fig_w = max(16, 2 + n_cols * 1.3)
    fig_h = max(4,  1 + n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    fig.suptitle(
        f"Filter performance summary  |  {folder}/{CHANNEL}\n"
        f"f0 = {f0:.5f} Hz  |  fs = {fs:.1f} Hz  |  "
        f"Attenuation = FFT peak reduction vs bandpassed at each harmonic",
        fontsize=11, fontweight='bold', y=0.98,
    )

    tbl = ax.table(
        cellText=rows_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    # Apply cell colors
    for r_idx, (row_t, row_c) in enumerate(zip(rows_text, rows_color)):
        for c_idx, color in enumerate(row_c):
            tbl[r_idx + 1, c_idx].set_facecolor(color)

    # Style header row
    for c_idx in range(n_cols):
        tbl[0, c_idx].set_facecolor('#1565C0')
        tbl[0, c_idx].set_text_props(color='white', fontweight='bold')

    # Wider first column
    tbl.auto_set_column_width(list(range(n_cols)))

    # ── color legend ──────────────────────────────────────────────────────────
    legend_items = [
        ('#A5D6A7', '>20 dB (excellent)'),
        ('#C8E6C9', '10-20 dB (good)'),
        ('#FFF9C4', '3-10 dB (marginal)'),
        ('#FFCDD2', '0-3 dB (poor)'),
        ('#EF9A9A', '<0 dB (amplification)'),
    ]
    legend_x = 0.01
    for color, label in legend_items:
        ax.add_patch(plt.Rectangle((legend_x, 0.01), 0.025, 0.035,
                                   transform=fig.transFigure,
                                   facecolor=color, edgecolor='grey',
                                   clip_on=False, linewidth=0.5))
        ax.text(legend_x + 0.030, 0.025, label,
                transform=fig.transFigure,
                fontsize=7.5, va='center')
        legend_x += 0.175

    fig.savefig("filter_comparison_table.png", dpi=150, bbox_inches='tight')
    print("Saved -> filter_comparison_table.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# METRICS (real data, no clean reference)
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(x_bp: np.ndarray, filtered: dict,
                  timings: dict, fs: float = 44100.0, f0: float = F0):
    """
    For real data without a clean reference, report:
      RMS     : overall amplitude of the filtered signal
      Atten   : FFT peak attenuation at f0 vs bandpassed input
      Atten2  : FFT peak attenuation at 2*f0
      Time    : wall-clock seconds for each filter
    """
    def peak_at(x, f_target, f_bw=1.5):
        fr, mag = compute_fft(x, fs)
        mask = (fr >= f_target - f_bw) & (fr <= f_target + f_bw)
        return float(np.max(mag[mask])) if mask.any() else np.nan

    peak_bp_1 = peak_at(x_bp, f0)
    peak_bp_2 = peak_at(x_bp, 2 * f0)

    print("\n-- Harmonic attenuation & timing -------------------------------------------")
    print(f"  {'Filter':<20}  {'RMS':>10}  {'Atten@f0':>12}  {'Atten@2xf0':>12}  {'Time':>8}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}")

    rms_bp = np.sqrt(np.mean(x_bp ** 2))
    print(f"  {'Bandpassed':<20}  {rms_bp:>10.6f}  {'(ref)':>12}  {'(ref)':>12}  {'---':>8}")

    for name, y in filtered.items():
        rms     = np.sqrt(np.mean(y ** 2))
        p1      = peak_at(y, f0)
        p2      = peak_at(y, 2 * f0)
        atten1  = 20 * np.log10(peak_bp_1 / (p1 + 1e-20))
        atten2  = 20 * np.log10(peak_bp_2 / (p2 + 1e-20))
        t_sec   = timings.get(name, float('nan'))
        print(f"  {name:<20}  {rms:>10.6f}  "
              f"{atten1:>+11.1f} dB  {atten2:>+11.1f} dB  {t_sec:>7.2f}s")

    print()
    print(f"  Atten@f0   : peak reduction at {f0:.5f} Hz vs bandpassed. Higher = better.")
    print(f"  Atten@2xf0 : peak reduction at {2*f0:.5f} Hz. Higher = better.")
    print(f"  RMS        : total signal energy. Should stay close to bandpassed.")
    print()
    print("  INTERPRETATION")
    print("  FIR Comb  RMS >> bandpassed: non-flat passband boosts mid-band EMG by")
    print("    up to +12 dB with filtfilt. High Atten partly from removing EMG signal")
    print("    around harmonics, not just noise. Do NOT use as-is for clean EMG.")
    print("  IIR Comb  RMS ~= bandpassed: flat passband between harmonics. Lower Atten")
    print("    than FIR Comb because the narrow 1.6 Hz notch only removes the powerline")
    print("    spike; residual EMG at those frequencies is correctly preserved.")
    print("  IIR Notch RMS ~= bandpassed: independent narrow notch (Q=30, ~1.7 Hz)")
    print("    per harmonic. Best passband fidelity, slowest per-harmonic cost at high fs.")
    print("  FIR Notch RMS < bandpassed : N capped at 4001 -> 88 Hz transition bands")
    print("    -> wide notch removes genuine EMG. Not suitable at fs=44100 Hz.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(folder: str = DEFAULT_FOLDER):
    # ── load ──────────────────────────────────────────────────────────────────
    t, x_raw, fs = load_recording(folder)
    f0 = F0

    # ── bandpass 6-500 Hz ─────────────────────────────────────────────────────
    print("Bandpass filtering 6-500 Hz ...")
    x_bp = bandpass_filter(x_raw, fs)

    # ── apply 4 filters ───────────────────────────────────────────────────────
    filter_fns = {
        'FIR Notch': lambda x: fir_notch_filter(x, fs, f0=f0),
        'IIR Notch': lambda x: iir_notch_filter(x, fs, f0=f0),
        'FIR Comb':  lambda x: fir_comb_filter(x,  fs, f0=f0),
        'IIR Comb':  lambda x: iir_comb_filter(x,  fs, f0=f0),
    }
    filtered = {}
    timings  = {}
    for name, fn in filter_fns.items():
        print(f"Applying {name} ...")
        t0 = time.perf_counter()
        filtered[name] = fn(x_bp)
        timings[name]  = time.perf_counter() - t0
        print(f"  done in {timings[name]:.2f} s")

    # ── FFTs ──────────────────────────────────────────────────────────────────
    print("Computing FFTs ...")
    ffts = {
        'raw':        compute_fft(x_raw, fs),
        'bandpassed': compute_fft(x_bp,  fs),
    }
    for name, y in filtered.items():
        ffts[name] = compute_fft(y, fs)

    # ── frequency responses (single stage, for plotting) ──────────────────────
    print("Computing single-stage frequency responses ...")
    responses = single_stage_responses(fs, f0=f0)

    # ── plot: main comparison figure ──────────────────────────────────────────
    print("Plotting main comparison figure ...")
    plot_demo(t, x_raw, x_bp, filtered, ffts, responses,
              fs=fs, f0=f0, folder=folder)

    # ── plot: full-signal overlay (Fig 2) ─────────────────────────────────────
    print("Plotting full-signal overlay ...")
    plot_overlay(t, x_raw, x_bp, filtered, ffts, fs=fs, f0=f0, folder=folder)

    # ── plot: time-domain subplots (Fig 3) ────────────────────────────────────
    print("Plotting time-domain subplots ...")
    plot_subplots_time(t, x_raw, x_bp, filtered, fs=fs, f0=f0, folder=folder)

    # ── plot: FFT subplots (Fig 4) ────────────────────────────────────────────
    print("Plotting FFT subplots ...")
    plot_subplots_fft(ffts, fs=fs, f0=f0, folder=folder)

    # ── plot: linear FFT subplots 0-800 Hz (Fig 5) ───────────────────────────
    print("Plotting linear-scale FFT subplots (0-800 Hz) ...")
    plot_subplots_fft_linear(ffts, fs=fs, f0=f0, folder=folder)

    # ── plot: performance summary table (Fig 6) ───────────────────────────────
    print("Plotting performance summary table ...")
    plot_performance_table(x_bp, filtered, timings, ffts, fs=fs, f0=f0,
                           folder=folder)

    # ── metrics ───────────────────────────────────────────────────────────────
    print_metrics(x_bp, filtered, timings, fs=fs, f0=f0)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    main(folder)
