"""
Notch filters NORMALIZED to unity passband -- ORIGINAL vs NORMALIZED, orders 1..10.

Companion to notch_filter_verification.py. Here we explicitly normalize both
notch filters so the full filter's passband peak is exactly 1, and check what
changes.

NORMALIZATION
-------------
Compute the full multi-harmonic filter's passband peak  P = max|H_full|  and
scale the output by 1/P, so the normalized filter has max|H_full| = 1.000 (no
frequency in the passband is amplified). This is the notch analogue of the comb
"patch", applied to the whole cascade so the result is exactly unity.

WHAT WE FIND
------------
* IIR Notch -- the iirnotch biquad already has |H| <= 1 with passband peak
  EXACTLY 1.0000 at every order, so 1/P = 1: normalization is a NO-OP. ORIGINAL
  and NORMALIZED rows are identical.

* FIR Notch -- the Hamming-window bandstop has a small passband ripple. Summed
  over the 10 harmonic sections and the cascade order it lifts the full passband
  peak slightly ABOVE 1 (P = 1.005 at order 1 -> 1.050 at order 10, i.e. up to
  +0.43 dB), and because the FIR is applied single-pass zero-phase that gain
  reaches the signal. Normalization scales it back to exactly 1.

So, unlike the comb (whose passband arched to +12 dB * order and truly needed the
patch), the notch barely needs normalizing: the IIR not at all, the FIR only by a
fraction of a dB. Normalization is a uniform level rescale -- it does NOT change
the mains-to-EMG ratio (attenuation and RMS shift together by 1/P).

APPLICATION NOTE
----------------
FIR notch is linear-phase -> applied ONCE (fftconvolve + delay trim); its gain is
|H|. IIR notch uses sosfiltfilt (forward+back) -> gain |H|^2, but since |H|=1
that is also 1. "pass|H|" is the full-filter passband peak that normalization
sets to 1.

RUN
---
    .venv/bin/python notch_filter_normalized.py [recording_folder]

Writes:
    notch_norm_results_table.html          -- ORIGINAL vs NORMALIZED, all orders
    notch_norm_freq_response.html          -- full-filter |H| (dB), orig vs norm
    notch_norm_freq_response_linear.html   -- passband zoom (unity check)
"""

from pathlib import Path
import sys
import time

import numpy as np
from scipy import signal as sig
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
RECORDINGS_DIR = Path("recordings")
DEFAULT_FOLDER = "emg_rec_20260625_092258"
CHANNEL        = "emg_line_L"
F0             = 49.97328
HARMONICS_MAX  = 800.0
Q              = 30.0
BW             = 5.0
MAX_FIR_TAPS   = 4001
ORDERS         = tuple(range(1, 11))
FILTERS        = ("FIR Notch", "IIR Notch")
VERSIONS       = ("ORIGINAL", "NORMALIZED")

C_FILTER = {"FIR Notch": "#E65100", "IIR Notch": "#4A148C"}


# --------------------------------------------------------------------------- #
# Loading + bandpass
# --------------------------------------------------------------------------- #
def load_recording(folder):
    path = RECORDINGS_DIR / folder / f"{CHANNEL}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["time"].astype(float)
    x = d["data"].astype(float)
    fs = 1.0 / (t[1] - t[0])
    print(f"Loaded : {path}")
    print(f"Samples: {len(x):,}   fs = {fs:.4f} Hz   duration = {t[-1]:.3f} s")
    return t, x, fs


def bandpass_filter(x, fs, low_hz=6.0, high_hz=500.0, order=4):
    sos = sig.butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, x)


# --------------------------------------------------------------------------- #
# Notch sections + full-filter response / normalization
# --------------------------------------------------------------------------- #
def _fir_taps(fs):
    n = int(8 * fs / BW)
    if n % 2 == 0:
        n += 1
    if n > MAX_FIR_TAPS:
        n = MAX_FIR_TAPS if MAX_FIR_TAPS % 2 else MAX_FIR_TAPS - 1
    return n


def base_section(kind, fc, fs):
    """One order-1 notch section at fc (unnormalized)."""
    if kind == "FIR Notch":
        n = _fir_taps(fs)
        b1 = sig.firwin(n, [max(fc - BW / 2, 0.5), min(fc + BW / 2, fs / 2 - 0.5)],
                        window="hamming", pass_zero=True, fs=fs)
        return b1, np.array([1.0])
    return sig.iirnotch(fc, Q, fs=fs)


def _power_conv(seq, n):
    out = np.array([1.0])
    for _ in range(n):
        out = np.convolve(out, seq)
    return out


def _apply_fir_zerophase(x, b):
    delay = (len(b) - 1) // 2
    y = sig.fftconvolve(x, b, mode="full")
    return y[delay:delay + len(x)]


def notch_filter(x, fs, kind, order):
    """Un-normalized notch at every harmonic (zero-phase)."""
    y = x.astype(float).copy()
    k = 1
    while k * F0 <= HARMONICS_MAX and k * F0 < fs / 2:
        b1, a1 = base_section(kind, k * F0, fs)
        if kind == "FIR Notch":
            y = _apply_fir_zerophase(y, _power_conv(b1, order))
        else:
            sos = np.tile(np.concatenate([b1, a1]), (order, 1))
            y = sig.sosfiltfilt(sos, y)
        k += 1
    return y


def full_response(kind, order, fs, normalize=False, worN=60000):
    """Full multi-harmonic filter response prod_k(H_k**order); optionally /peak."""
    w = H = None
    k = 1
    while k * F0 <= HARMONICS_MAX and k * F0 < fs / 2:
        b, a = base_section(kind, k * F0, fs)
        wk, Hk = sig.freqz(b, a, worN=worN, fs=fs)
        Hk = Hk ** order
        w, H = (wk, Hk) if H is None else (w, H * Hk)
        k += 1
    if normalize:
        H = H / float(np.max(np.abs(H)))
    return w, H


def full_passband_peak(kind, order, fs):
    _, H = full_response(kind, order, fs, normalize=False)
    return float(np.max(np.abs(H)))


def rfft_mag(x, fs):
    N = len(x)
    fr = np.fft.rfftfreq(N, 1.0 / fs)
    return fr, np.abs(np.fft.rfft(x)) * (2.0 / N)


def peak_near(fr, mag, f_target, f_bw=1.5):
    m = (fr >= f_target - f_bw) & (fr <= f_target + f_bw)
    return float(np.max(mag[m])) if m.any() else np.nan


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(folder=DEFAULT_FOLDER):
    t, x_raw, fs = load_recording(folder)
    x_bp = bandpass_filter(x_raw, fs)
    rms_ref = rms(x_bp)
    fr_ref, mag_ref = rfft_mag(x_bp, fs)
    ref_f0 = peak_near(fr_ref, mag_ref, F0)
    ref_2f0 = peak_near(fr_ref, mag_ref, 2 * F0)

    print(f"\nNormalizing full-filter passband peak to 1  (scale = 1/max|H_full|)")
    print(f"RMS bandpassed (reference) = {rms_ref:.6f} V\n")

    rows = []
    for kind in FILTERS:
        for order in ORDERS:
            peak = full_passband_peak(kind, order, fs)     # un-normalized peak P
            t0 = time.perf_counter()
            y = notch_filter(x_bp, fs, kind, order)        # filter once
            dt_ms = (time.perf_counter() - t0) * 1e3
            for version in VERSIONS:
                yv = y if version == "ORIGINAL" else y / peak
                pk = peak if version == "ORIGINAL" else 1.0
                fr, mag = rfft_mag(yv, fs)
                red_f0  = ref_f0  / (peak_near(fr, mag, F0)     + 1e-20)
                red_2f0 = ref_2f0 / (peak_near(fr, mag, 2 * F0) + 1e-20)
                r = rms(yv)
                rows.append(dict(
                    filter=kind, version=version, order=order,
                    peak=pk, peak_db=20 * np.log10(pk),
                    att_f0=20 * np.log10(red_f0),   att_f0_lin=red_f0,
                    att_2f0=20 * np.log10(red_2f0), att_2f0_lin=red_2f0,
                    rms=r, xref=r / rms_ref, time_ms=dt_ms))

    _print_table(rows)
    _summarize(rows)

    plot_results_table_plotly(rows, folder)
    plot_freq_response_plotly(fs, scale="db")
    plot_freq_response_plotly(fs, scale="linear")


def _print_table(rows):
    header = (f"{'filter':<11}{'version':<12}{'order':>6}{'pass|H|':>9}"
              f"{'pass dB':>9}{'att50dB':>9}{'att50x':>9}{'att100dB':>10}"
              f"{'att100x':>9}{'RMS(V)':>11}{'x ref':>8}{'ms':>8}")
    for kind in FILTERS:
        print("=" * len(header)); print(kind); print("=" * len(header))
        print(header); print("-" * len(header))
        for version in VERSIONS:
            for r in [r for r in rows if r["filter"] == kind
                      and r["version"] == version]:
                print(f"{r['filter']:<11}{r['version']:<12}{r['order']:>6}"
                      f"{r['peak']:>9.4f}{r['peak_db']:>+9.2f}"
                      f"{r['att_f0']:>+9.1f}{r['att_f0_lin']:>9.3g}"
                      f"{r['att_2f0']:>+10.1f}{r['att_2f0_lin']:>9.3g}"
                      f"{r['rms']:>11.6f}{r['xref']:>8.2f}{r['time_ms']:>8.1f}")
            print("-" * len(header))
        print()


def _summarize(rows):
    def get(kind, version, order, key):
        for r in rows:
            if (r["filter"], r["version"], r["order"]) == (kind, version, order):
                return r[key]
    print("RESULT OF NORMALIZING TO 1")
    p1, p10 = get("FIR Notch", "ORIGINAL", 1, "peak"), get("FIR Notch", "ORIGINAL", 10, "peak")
    print(f"  FIR Notch passband peak (ORIGINAL): {p1:.4f} (+{20*np.log10(p1):.2f} dB) "
          f"@order1  ->  {p10:.4f} (+{20*np.log10(p10):.2f} dB) @order10")
    print(f"           after NORMALIZE: exactly 1.0000 at every order "
          f"(scale 1/P; @order10 that is x{1/p10:.4f}).")
    print(f"  IIR Notch passband peak: {get('IIR Notch','ORIGINAL',10,'peak'):.4f} "
          "at every order -> normalization is a NO-OP (ORIGINAL == NORMALIZED).")
    print("  att@50 / RMS move together by the 1/P scale -> mains-to-EMG ratio")
    print("  unchanged. Conclusion: notch needs (almost) no normalization,")
    print("  unlike the comb whose passband arched to +12 dB * order.\n")


def plot_results_table_plotly(rows, folder):
    cols = ["Filter", "Version", "Order", "pass|H|", "pass dB",
            "Att@50 dB", "Att@50 ×", "Att@100 dB", "Att@100 ×",
            "RMS (V)", "x ref", "Time (ms)"]

    def fmt(r):
        return [r["filter"], r["version"], str(r["order"]),
                f"{r['peak']:.4f}", f"{r['peak_db']:+.2f}",
                f"{r['att_f0']:+.1f}", f"{r['att_f0_lin']:.3g}",
                f"{r['att_2f0']:+.1f}", f"{r['att_2f0_lin']:.3g}",
                f"{r['rms']:.6f}", f"{r['xref']:.2f}", f"{r['time_ms']:.1f}"]

    # order rows filter-major, version-major so ORIGINAL & NORMALIZED are grouped
    ordered = [r for kind in FILTERS for version in VERSIONS
               for r in rows if r["filter"] == kind and r["version"] == version]
    text = [fmt(r) for r in ordered]
    row_color = []
    for r in ordered:
        if r["version"] == "ORIGINAL" and r["peak"] > 1.005:
            row_color.append("#FFCDD2")            # original passband > 1
        elif r["version"] == "NORMALIZED":
            row_color.append("#C8E6C9")            # normalized to exactly 1
        else:
            row_color.append("#FFFFFF")
    columns = list(zip(*text))
    fill = [list(row_color) for _ in cols]

    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in cols],
                    fill_color="#1565C0", font=dict(color="white", size=11),
                    align="center", height=30),
        cells=dict(values=columns, fill_color=fill, align="center",
                   font=dict(size=10), height=22)))
    fig.update_layout(
        title=("Notch filters normalized to unity passband -- ORIGINAL vs "
               f"NORMALIZED  |  {folder}/{CHANNEL}"
               "<br><sup>green = NORMALIZED (pass|H| = 1.0000); pink = ORIGINAL "
               "with passband peak &gt; 1 (FIR only). IIR rows are identical "
               "(already unity). att/RMS shift only by the uniform 1/P scale.</sup>"),
        template="plotly_white", height=900, margin=dict(t=95, b=20))
    _show_or_note(fig, "notch_norm_results_table.html")


def plot_freq_response_plotly(fs, scale="db"):
    """Full-filter |H| vs frequency, ORIGINAL (solid) vs NORMALIZED (dashed),
    orders 1 & 10. Linear view zooms the passband to make the unity check clear;
    dB view shows the full notch depth."""
    linear = (scale == "linear")
    fmax = 3.2 * F0
    show_orders = (1, 10)
    fig = make_subplots(rows=1, cols=2, subplot_titles=FILTERS,
                        horizontal_spacing=0.09)
    unity = 1.0 if linear else 0.0

    for col, kind in enumerate(FILTERS, start=1):
        for order in show_orders:
            shade = 0.5 if order == show_orders[0] else 1.0
            for version, norm, dash in (("ORIGINAL", False, "solid"),
                                        ("NORMALIZED", True, "dash")):
                w, H = full_response(kind, order, fs, normalize=norm, worN=120000)
                m = w <= fmax
                g = np.abs(H[m])
                y = g if linear else 20 * np.log10(g + 1e-12)
                fig.add_trace(go.Scatter(
                    x=w[m], y=y, mode="lines",
                    line=dict(color=_scale_color(C_FILTER[kind], shade),
                              width=1.7, dash=dash),
                    name=f"order {order} {version}",
                    showlegend=(col == 1)), row=1, col=col)
        fig.add_hline(y=unity, line=dict(color="grey", dash="dot", width=1),
                      row=1, col=col)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[0, fmax],
                         row=1, col=col)
        # linear zooms the passband (0.85..1.12); dB shows the full notch
        fig.update_yaxes(title_text="|H| (linear)" if linear else "|H| (dB)",
                         range=[0.85, 1.12] if linear else [-90, 6], row=1, col=col)

    note = ("linear |H| PASSBAND ZOOM (0.85..1.12): grey = unity. FIR ORIGINAL "
            "rides ABOVE 1 (up to +0.4 dB at order 10); NORMALIZED (dashed) sits "
            "on 1. IIR: solid & dashed both on 1. Notches plunge below the frame."
            if linear else
            "dB |H|: 0 dB = unity passband; notches dip down. FIR orig vs norm "
            "differ by a fraction of a dB in the passband; IIR curves coincide.")
    fig.update_layout(
        title=(f"Notch full-filter response -- ORIGINAL vs NORMALIZED "
               f"({'linear passband zoom' if linear else 'dB'}), orders 1 & 10"
               f"<br><sup>{note}</sup>"),
        template="plotly_white", height=520, hovermode="x unified")
    out = ("notch_norm_freq_response_linear.html" if linear
           else "notch_norm_freq_response.html")
    _show_or_note(fig, out)


def _scale_color(hex_color, f):
    """Lighten a hex color toward white by factor (0..1); 1.0 = original."""
    c = np.array([int(hex_color[i:i+2], 16) for i in (1, 3, 5)], float)
    c = c + (255 - c) * (1 - f)
    return "#%02X%02X%02X" % tuple(c.astype(int))


def _show_or_note(fig, out):
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved interactive plot -> {out}")
    try:
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    if not (RECORDINGS_DIR / folder / f"{CHANNEL}.csv").exists():
        avail = sorted(d.name for d in RECORDINGS_DIR.iterdir()
                       if (d / f"{CHANNEL}.csv").exists())
        if not avail:
            sys.exit(f"No recordings with {CHANNEL}.csv under {RECORDINGS_DIR}/")
        folder = DEFAULT_FOLDER if DEFAULT_FOLDER in avail else avail[0]
        print(f"(folder not given/found; using '{folder}')")
    main(folder)
