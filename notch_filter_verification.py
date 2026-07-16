"""
Notch-filter order analysis (FIR vs IIR), orders 1..10 -- the notch counterpart
of comb_filter_verification.py.

WHAT "ORDER" MEANS
------------------
Order N = the per-harmonic notch section cascaded N times:

    IIR Notch   [iirnotch(f0, Q)]^N     -- biquad (2 poles/2 zeros) ^ N
    FIR Notch   [firwin bandstop]^N     -- Hamming-window bandstop ^ N

A separate section is placed at every mains harmonic up to HARMONICS_MAX, so the
full filter is  (cascade over harmonics) of (cascade of N per harmonic).  All
applied zero-phase.

WHY THERE IS NO "PATCHED" VARIANT (unlike the comb)
---------------------------------------------------
The comb needed an amplitude patch because its passband arches ABOVE unity (up
to +12 dB * order) and filtfilt squared that into signal amplification. A notch
does not have that problem: every notch section has a passband gain of <= 1 by
construction (IIR biquad = exactly 1.000; FIR Hamming bandstop ~= 1.006), so the
cascade stays at unity for every order and filtfilt cannot inflate the signal.
Hence RMS(filtered) <= RMS(bandpassed) at all orders and no normalization is
needed -- so this file analyses FIR Notch vs IIR Notch x order only.

NOTCH vs COMB, A KEY DIFFERENCE
------------------------------
The notch sections are tuned to the EXACT measured mains f0 (49.973 Hz), whereas
the comb's nulls are locked to the fs/M grid (50.000 Hz). So the notch sits right
on the interference and rejects it hard even at low order.

TWO PRACTICAL POINTS
--------------------
* FIR Notch needs ~8*fs/bw taps (70k at fs=44.1 kHz, bw=5 Hz) -- impractical, and
  filtfilt would allocate a huge zi matrix. We cap at MAX_FIR_TAPS and apply it
  zero-phase via fftconvolve + group-delay trim (a symmetric FIR has constant
  delay (len-1)/2). The cap makes the FIR notch very WIDE (~88 Hz), so it removes
  real EMG -- that is the headline FIR-notch limitation.
* IIR Notch is a biquad: 3 taps per section, cascaded to 2N+1 -- tiny and fast.

RUN
---
    .venv/bin/python notch_filter_verification.py [recording_folder]

Prints a per-filter x order table (passband peak, filtfilt gain dB+linear,
harmonic attenuation dB+linear, RMS vs reference, filtfilt/fftconvolve apply
time) and writes interactive Plotly HTML files:

    notch_filter_results_table.html          -- color-coded results + timing;
                                                recommended order highlighted
    notch_filter_freq_response.html          -- notch gain in dB (log) vs
                                                frequency, a line per order
    notch_filter_freq_response_linear.html   -- same on a LINEAR axis (notch
                                                floor reaches ~0)
    notch_filter_time_overlay.html           -- bandpass reference overlaid with
                                                the filter output at each order
"""

from pathlib import Path
import sys
import time

import numpy as np
from scipy import signal as sig
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Constants (kept consistent with filter_comparison_notebook.ipynb)
# --------------------------------------------------------------------------- #
RECORDINGS_DIR = Path("recordings")
DEFAULT_FOLDER = "emg_rec_20260625_092258"
CHANNEL        = "emg_line_L"
F0             = 49.97328   # Hz -- measured mains frequency (notch tuned here)
HARMONICS_MAX  = 800.0      # Hz -- notch every harmonic up to this
Q              = 30.0       # IIR notch quality factor (~1.7 Hz wide at 50 Hz)
BW             = 5.0        # Hz -- FIR notch target stopband width
MAX_FIR_TAPS   = 4001       # cap so the FIR notch stays finite at audio rates
ORDERS         = tuple(range(1, 11))   # cascade orders 1..10


def _shades(light, dark, n):
    """n hex colors interpolated light->dark (per-order gradient)."""
    lo = np.array([int(light[i:i+2], 16) for i in (1, 3, 5)], float)
    hi = np.array([int(dark[i:i+2], 16)  for i in (1, 3, 5)], float)
    return ["#%02X%02X%02X" % tuple(np.round(lo + (hi - lo) * f).astype(int))
            for f in np.linspace(0, 1, n)]


C_BANDPASS  = "#555555"
ORDER_SHADE = {                                # FIR notch orange, IIR purple
    "FIR Notch": _shades("#FFCC80", "#E65100", len(ORDERS)),
    "IIR Notch": _shades("#CE93D8", "#4A148C", len(ORDERS)),
}
FILTERS = ("FIR Notch", "IIR Notch")


# --------------------------------------------------------------------------- #
# Loading + bandpass (same pipeline as the notebook)
# --------------------------------------------------------------------------- #
def load_recording(folder):
    """Load one channel CSV -> (t, x, fs). fs from the actual sample interval."""
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
    """Zero-phase Butterworth bandpass -- the reference signal for comparison."""
    sos = sig.butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, x)


# --------------------------------------------------------------------------- #
# Notch sections, order cascade, and zero-phase application
# --------------------------------------------------------------------------- #
def _power_conv(seq, n):
    """Polynomial power seq^n by repeated convolution (the order-n cascade)."""
    out = np.array([1.0])
    for _ in range(n):
        out = np.convolve(out, seq)
    return out


def _fir_taps(fs):
    """Odd tap count for the FIR notch, capped at MAX_FIR_TAPS."""
    n = int(8 * fs / BW)
    if n % 2 == 0:
        n += 1
    if n > MAX_FIR_TAPS:
        n = MAX_FIR_TAPS if MAX_FIR_TAPS % 2 else MAX_FIR_TAPS - 1
    return n


def base_section(kind, fc, fs):
    """(b, a) for ONE order-1 notch section at fc."""
    if kind == "FIR Notch":
        n = _fir_taps(fs)
        b1 = sig.firwin(n, [max(fc - BW / 2, 0.5), min(fc + BW / 2, fs / 2 - 0.5)],
                        window="hamming", pass_zero=True, fs=fs)
        return b1, np.array([1.0])
    return sig.iirnotch(fc, Q, fs=fs)


def section_response(kind, order, fc, fs, worN):
    """
    Frequency response of the order-N section = H_base ** order.

    Computed as a power of the WELL-CONDITIONED base biquad response -- never
    expand the IIR denominator into a degree-2N polynomial: its N-fold poles sit
    right on |z|=1, so a direct-form evaluation explodes (pass|H| -> hundreds,
    RMS -> 1e130). Cascading is done with second-order sections instead.
    """
    w, H1 = sig.freqz(*base_section(kind, fc, fs), worN=worN, fs=fs)
    return w, H1 ** order


def _apply_fir_zerophase(x, b):
    """Zero-phase FIR via fftconvolve + constant-group-delay trim (memory-safe;
    filtfilt would allocate a huge zi matrix for a long FIR). FIR has no poles,
    so raising it to a power by convolution stays well-conditioned."""
    delay = (len(b) - 1) // 2
    y = sig.fftconvolve(x, b, mode="full")
    return y[delay:delay + len(x)]


def notch_filter(x, fs, kind, order):
    """Apply the notch at every harmonic up to HARMONICS_MAX, zero-phase.

    IIR order cascade uses sosfiltfilt (the section repeated `order` times as
    second-order sections) -- numerically stable, unlike an expanded polynomial.
    """
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


def passband_peak(kind, order, fc, fs, npts=100000):
    """Peak |H| of the order-N section over 0..Nyquist (~1 for a notch)."""
    _, H1 = sig.freqz(*base_section(kind, fc, fs), worN=npts)
    return float(np.max(np.abs(H1)) ** order)


def rfft_mag(x, fs):
    N = len(x)
    fr = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(np.fft.rfft(x)) * (2.0 / N)
    return fr, mag


def peak_near(fr, mag, f_target, f_bw=1.5):
    m = (fr >= f_target - f_bw) & (fr <= f_target + f_bw)
    return float(np.max(mag[m])) if m.any() else np.nan


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def main(folder=DEFAULT_FOLDER):
    t, x_raw, fs = load_recording(folder)
    x_bp = bandpass_filter(x_raw, fs)
    rms_ref = rms(x_bp)
    fr_ref, mag_ref = rfft_mag(x_bp, fs)
    ref_f0 = peak_near(fr_ref, mag_ref, F0)
    ref_2f0 = peak_near(fr_ref, mag_ref, 2 * F0)

    n_harm = sum(1 for k in range(1, 999) if k * F0 <= HARMONICS_MAX and k * F0 < fs / 2)
    print(f"\nQ = {Q}   bw(FIR) = {BW} Hz   harmonics notched = {n_harm}   "
          f"orders = {list(ORDERS)}")
    print(f"RMS bandpassed (reference) = {rms_ref:.6f} V")
    print(f"Reference mains peaks:  @f0={ref_f0:.5f} V   @2f0={ref_2f0:.5f} V")

    filtered = {}   # (kind, order) -> y
    rows = []

    header = (f"{'filter':<11}{'order':>6}{'taps':>7}{'pass|H|':>9}"
              f"{'flt dB':>8}{'flt lin':>9}"
              f"{'att50dB':>8}{'att50x':>10}{'att100dB':>9}{'att100x':>10}"
              f"{'RMS(V)':>11}{'x ref':>8}{'ms':>8}")

    for kind in FILTERS:
        print("\n" + "=" * len(header))
        print(kind)
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for order in ORDERS:
            if kind == "FIR Notch":
                b1, _ = base_section(kind, F0, fs)
                taps = order * (len(b1) - 1) + 1        # FIR cascade length
            else:
                taps = 2 * order + 1                    # IIR denominator order+1
            peak = passband_peak(kind, order, F0, fs)

            t0 = time.perf_counter()
            y = notch_filter(x_bp, fs, kind, order)
            dt_ms = (time.perf_counter() - t0) * 1e3
            filtered[(kind, order)] = y

            fr, mag = rfft_mag(y, fs)
            red_f0  = ref_f0  / (peak_near(fr, mag, F0)     + 1e-20)
            red_2f0 = ref_2f0 / (peak_near(fr, mag, 2 * F0) + 1e-20)
            flt_lin = peak ** 2
            r = rms(y)
            rows.append(dict(
                filter=kind, order=order, taps=taps,
                peak=peak, flt_db=20 * np.log10(flt_lin), flt_lin=flt_lin,
                att_f0=20 * np.log10(red_f0),   att_f0_lin=red_f0,
                att_2f0=20 * np.log10(red_2f0), att_2f0_lin=red_2f0,
                rms=r, xref=r / rms_ref, time_ms=dt_ms))
            print(f"{kind:<11}{order:>6}{taps:>7}{peak:>9.4f}"
                  f"{20*np.log10(flt_lin):>+8.2f}{flt_lin:>9.4f}"
                  f"{20*np.log10(red_f0):>+8.1f}{red_f0:>10.3g}"
                  f"{20*np.log10(red_2f0):>+9.1f}{red_2f0:>10.3g}"
                  f"{r:>11.6f}{r/rms_ref:>8.2f}{dt_ms:>8.1f}")
        print("-" * len(header))

    best = recommend_orders(rows)
    print("\nHOW ORDER IMPACTS FILTRATION (notch)")
    print("  * Passband stays at unity (pass|H| ~= 1) for EVERY order -> no")
    print("    amplitude inflation; 'x ref' stays <= 1 (no patch needed).")
    print("  * Higher order deepens/steepens the notch -> more mains rejection,")
    print("    but returns diminish because the notch is already tuned to f0.")
    print("  * IIR notch: 3 taps/section (2N+1 cascaded) -> tiny, fast, narrow;")
    print("    preserves EMG. FIR notch: capped-tap -> ~88 Hz WIDE, removes EMG.")
    print("  * Cost grows ~linearly with order.")
    print(f"\n  RECOMMENDED  ->  FIR Notch: order {best['FIR Notch']}   "
          f"IIR Notch: order {best['IIR Notch']}   (reasoning in the table)\n")

    # ------------------------------- plots --------------------------------- #
    plot_results_table_plotly(rows, best, folder)
    plot_freq_response_plotly(fs, scale="db")
    plot_freq_response_plotly(fs, scale="linear")
    plot_time_overlay_plotly(t, x_bp, rms_ref, filtered, folder)


def recommend_orders(rows, target_att_db=20.0):
    """
    Best order per filter: the SMALLEST order whose mains rejection (att@f0)
    reaches target_att_db. Beyond that, extra order only costs taps/time/ringing
    (and, for the wide FIR notch, more EMG loss) for little added rejection.
    """
    best = {}
    for kind in FILTERS:
        rs = sorted((r for r in rows if r["filter"] == kind),
                    key=lambda r: r["order"])
        reach = [r for r in rs if r["att_f0"] >= target_att_db]
        best[kind] = (reach[0] if reach
                      else max(rs, key=lambda r: r["att_f0"]))["order"]
    return best


def plot_results_table_plotly(rows, best, folder):
    """Color-coded results table (both filters x orders + dB/linear + timing)."""
    cols = ["Filter", "Order", "Taps", "pass|H|",
            "filtfilt dB", "filtfilt |H|²",
            "Att@50 dB", "Att@50 ×", "Att@100 dB", "Att@100 ×",
            "RMS (V)", "x ref", "Time (ms)"]

    def fmt(r):
        return [r["filter"], str(r["order"]), str(r["taps"]),
                f"{r['peak']:.4f}",
                f"{r['flt_db']:+.2f}", f"{r['flt_lin']:.4f}",
                f"{r['att_f0']:+.1f}", f"{r['att_f0_lin']:.3g}",
                f"{r['att_2f0']:+.1f}", f"{r['att_2f0_lin']:.3g}",
                f"{r['rms']:.6f}", f"{r['xref']:.2f}", f"{r['time_ms']:.1f}"]

    text = [fmt(r) for r in rows]
    row_color = ["#A5D6A7" if r["order"] == best[r["filter"]] else "#FFFFFF"
                 for r in rows]
    columns = list(zip(*text))
    fill = [list(row_color) for _ in cols]

    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in cols],
                    fill_color="#1565C0", font=dict(color="white", size=11),
                    align="center", height=30),
        cells=dict(values=columns, fill_color=fill, align="center",
                   font=dict(size=10), height=22)))
    fig.update_layout(
        title=(f"Notch-filter results -- orders {ORDERS[0]}..{ORDERS[-1]}  |  "
               f"{folder}/{CHANNEL}"
               f"<br><sup>green = recommended (FIR order {best['FIR Notch']}, "
               f"IIR order {best['IIR Notch']}). pass|H| ~= 1 at every order "
               "(unity passband -> no amplitude inflation). Time = apply time "
               "on this signal.</sup>"),
        template="plotly_white", height=760, margin=dict(t=90, b=20))
    _show_or_note(fig, "notch_filter_results_table.html")


def _zoom(kind):
    """Per-filter x-range: FIR notch is wide (~88 Hz), IIR is narrow (~1.7 Hz)."""
    return [max(0.0, F0 - 45), F0 + 45] if kind == "FIR Notch" else [F0 - 8, F0 + 8]


def plot_freq_response_plotly(fs, scale="db"):
    """
    Single-notch (at f0) gain vs frequency: 1x2 = FIR | IIR, a line per order.
    scale="db"     -> y = 20*log10(|H|^2) filtfilt gain; unity = 0 dB.
    scale="linear" -> y = |H|^2; notch floor reaches ~0; unity = 1.0.
    """
    linear = (scale == "linear")
    fig = make_subplots(rows=1, cols=2, subplot_titles=FILTERS,
                        horizontal_spacing=0.09)
    unity = 1.0 if linear else 0.0
    y_title = "filtfilt gain |H|^2 (linear)" if linear else "filtfilt gain (dB)"

    for col, kind in enumerate(FILTERS, start=1):
        lo, hi = _zoom(kind)
        for order in ORDERS:
            w, H = section_response(kind, order, F0, fs, 200000)
            m = (w >= lo) & (w <= hi)
            g = np.abs(H[m]) ** 2
            y = g if linear else 20 * np.log10(g + 1e-12)
            fig.add_trace(go.Scatter(
                x=w[m], y=y, mode="lines",
                line=dict(color=ORDER_SHADE[kind][order - 1], width=1.8),
                name=f"order {order}", legendgroup=f"order{order}",
                showlegend=(col == 1)), row=1, col=col)
        fig.add_hline(y=unity, line=dict(color="grey", dash="dot", width=1),
                      row=1, col=col)
        fig.add_vline(x=F0, line=dict(color="red", dash="dot", width=0.7),
                      opacity=0.4, row=1, col=col)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[lo, hi], row=1, col=col)
        fig.update_yaxes(title_text=y_title,
                         range=[0, 1.15] if linear else [-100, 5], row=1, col=col)

    note = ("linear magnitude -- notch floor reaches ~0; grey line = unity (1.0)"
            if linear else
            "0 dB grey line = unity passband; the notch deepens with order")
    fig.update_layout(
        title=(f"Notch filters: single-notch gain at f0, orders 1..10 "
               f"({'linear' if linear else 'dB / log'} scale)"
               f"<br><sup>{note} &nbsp;|&nbsp; red dotted = f0={F0:.3f} Hz &nbsp;|"
               "&nbsp; FIR panel is wide (capped taps), IIR panel is narrow "
               "(Q=30)</sup>"),
        template="plotly_white", height=520, hovermode="x unified")
    out = ("notch_filter_freq_response_linear.html" if linear
           else "notch_filter_freq_response.html")
    _show_or_note(fig, out)


def _envelope(t, x, n_bins=4000):
    """Peak-preserving decimation (keep min & max of each time-bin)."""
    n = len(x)
    if n <= 2 * n_bins:
        return t, x
    bin_len = n // n_bins
    usable = bin_len * n_bins
    idx = np.arange(usable).reshape(n_bins, bin_len)
    xb = x[:usable].reshape(n_bins, bin_len)
    rows = np.arange(n_bins)
    keep = np.union1d(idx[rows, np.argmin(xb, axis=1)],
                      idx[rows, np.argmax(xb, axis=1)])
    if usable < n:
        keep = np.union1d(keep, np.arange(usable, n))
    return t[keep], x[keep]


def plot_time_overlay_plotly(t, x_bp, rms_ref, filtered, folder):
    """1x2 = FIR | IIR: bandpass reference (grey) overlaid with each order."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=FILTERS,
                        shared_yaxes=True, horizontal_spacing=0.06)
    t_ref, y_ref = _envelope(t, x_bp)
    for col, kind in enumerate(FILTERS, start=1):
        fig.add_trace(go.Scattergl(
            x=t_ref, y=y_ref, mode="lines",
            line=dict(color=C_BANDPASS, width=1),
            name="Bandpassed (ref)", legendgroup="ref",
            showlegend=(col == 1)), row=1, col=col)
        for order in ORDERS:
            t_f, y_f = _envelope(t, filtered[(kind, order)])
            xr = rms(filtered[(kind, order)]) / rms_ref
            fig.add_trace(go.Scattergl(
                x=t_f, y=y_f, mode="lines",
                line=dict(color=ORDER_SHADE[kind][order - 1], width=1),
                name=f"order {order} (x{xr:.2f})",
                legendgroup=f"{kind}o{order}"), row=1, col=col)
        fig.update_xaxes(title_text="Time (s)", row=1, col=col)
    fig.update_yaxes(title_text="Amplitude (V)", row=1, col=1)
    fig.update_layout(
        title=(f"Notch: time-domain amplitude vs order 1..10  |  "
               f"{folder}/{CHANNEL}"
               "<br><sup>grey = bandpassed reference; legend shows RMS x reference"
               " (notch never inflates -> all orders sit at/under the reference)"
               "</sup>"),
        template="plotly_white", height=560, hovermode="x unified")
    _show_or_note(fig, "notch_filter_time_overlay.html")


def _show_or_note(fig, out):
    """Write an interactive HTML file; also try to open it in a browser."""
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
