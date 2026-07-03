"""
Comb-filter amplitude verification: ORIGINAL vs PATCHED, orders 1..5.

Two things are verified here:

1. AMPLITUDE (original vs patched)
   Both comb filters amplify signal energy that sits BETWEEN the powerline
   harmonics, because their passband arches above unity gain:

       FIR comb   H(z) = 1 - z^-M                  -> passband peak = 2
       IIR comb   H(z) = (1 - z^-M)/(1 - r_m z^-M) -> passband peak = 2/(1+r_m)

   `filtfilt` (zero-phase) applies the filter twice, so the gain that reaches
   the signal is the peak SQUARED (FIR: 4 = +12 dB; IIR at r_m=0.9: +0.9 dB).

   THE PATCH scales the feed-forward (numerator) coefficients so the passband
   peak of ONE section becomes exactly 1.0, leaving notch depth/position alone:

       FIR comb   b *= 1/2            IIR comb   b *= (1+r_m)/2

2. ORDER (1..5)
   "Order" = number of cascaded identical comb sections. Order N is:

       FIR comb   H(z) = (1 - z^-M)^N
       IIR comb   H(z) = [(1 - z^-M)/(1 - r_m z^-M)]^N

   Higher order -> deeper, steeper-skirted notches (better harmonic rejection),
   BUT the passband arch is raised to the Nth power too, so the between-harmonic
   boost EXPLODES for the ORIGINAL design:

       FIR original filtfilt passband peak = 4^N   (order 5 -> 1024x, +60 dB!)
       IIR original filtfilt passband peak = (2/(1+r_m))^(2N)

   The PATCHED design keeps the passband peak at 1^N = 1 for EVERY order, so you
   get the sharper notches of a high-order comb WITHOUT the amplitude blow-up.
   Stability is unaffected by order (poles keep radius r_m^(1/M) < 1, just with
   multiplicity N).

RUN
---
    .venv/bin/python comb_filter_verification.py [recording_folder]

Prints a table (passband peak, filtfilt dB, harmonic attenuation, RMS vs
reference) for every filter x version x order, and writes two interactive
Plotly HTML files (open in any browser; drag to zoom, click legend to toggle):

    comb_filter_freq_response.html         -- filtfilt gain in dB (log) vs
                                              frequency, one panel per filter x
                                              version, a line per order 1..5
    comb_filter_freq_response_linear.html  -- same characteristic on a LINEAR
                                              magnitude axis, so the notches
                                              visibly reach exactly 0 and the
                                              passband arch heights are true to
                                              scale
    comb_filter_time_overlay.html   -- bandpass reference overlaid with the
                                       filter output at each order, so the
                                       time-domain amplitude is directly
                                       comparable across orders.
"""

from pathlib import Path
import sys

import numpy as np
from scipy import signal as sig
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Constants (kept identical to filter_comparison_notebook.ipynb)
# --------------------------------------------------------------------------- #
RECORDINGS_DIR = Path("recordings")
DEFAULT_FOLDER = "emg_rec_20260625_092258"
CHANNEL        = "emg_line_L"
F0             = 49.97328   # Hz -- measured mains frequency
R_M            = 0.9        # IIR comb feedback coefficient
ORDERS         = (1, 2, 3, 4, 5)

# Colors: grey reference + a light->dark gradient per filter, indexed by order
C_BANDPASS  = "#555555"
ORDER_SHADE = {
    "FIR comb": ["#A5D6A7", "#66BB6A", "#43A047", "#2E7D32", "#1B5E20"],
    "IIR comb": ["#EF9A9A", "#E57373", "#EF5350", "#C62828", "#8E0000"],
}


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
# The comb filters: FIR/IIR x original/patched x order 1..5
# --------------------------------------------------------------------------- #
def comb_coeffs(fs, order, f0=F0, r_m=R_M, normalize=False):
    """
    Return (b, a) for FIR and IIR combs at the given cascade `order`.

    order   -> section cascaded this many times: FIR (1-z^-M)^order,
               IIR [(1-z^-M)/(1-r_m z^-M)]^order.
    normalize=False -> ORIGINAL coefficients (passband arches above 1).
    normalize=True  -> PATCHED: each section's numerator scaled so a single
                       section's passband peak == 1 (FIR *1/2, IIR *(1+r_m)/2);
                       the cascade therefore stays at 1 for every order.
    """
    M = int(round(fs / f0))

    # one section
    b_fir1 = np.zeros(M + 1); b_fir1[0], b_fir1[M] = 1.0, -1.0
    b_iir1 = b_fir1.copy()
    a_iir1 = np.zeros(M + 1); a_iir1[0], a_iir1[M] = 1.0, -r_m
    if normalize:
        b_fir1 *= 0.5
        b_iir1 *= (1.0 + r_m) / 2.0

    # cascade = repeated polynomial multiplication (convolution)
    b_fir = np.array([1.0])
    b_iir = np.array([1.0])
    a_iir = np.array([1.0])
    for _ in range(order):
        b_fir = np.convolve(b_fir, b_fir1)
        b_iir = np.convolve(b_iir, b_iir1)
        a_iir = np.convolve(a_iir, a_iir1)

    return {
        "FIR comb": (b_fir, np.array([1.0])),
        "IIR comb": (b_iir, a_iir),
    }, M


def passband_peak(b, a, npts=200000):
    """Peak |H| over 0..Nyquist (single pass). filtfilt peak = this squared."""
    _, H = sig.freqz(b, a, worN=npts)
    return float(np.max(np.abs(H)))


def rfft_mag(x, fs):
    """Single-sided amplitude spectrum (freqs, magnitude)."""
    N = len(x)
    fr = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(np.fft.rfft(x)) * (2.0 / N)
    return fr, mag


def peak_near(fr, mag, f_target, f_bw=1.5):
    """Largest spectral magnitude within +/- f_bw Hz of f_target."""
    m = (fr >= f_target - f_bw) & (fr <= f_target + f_bw)
    return float(np.max(mag[m])) if m.any() else np.nan


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# --------------------------------------------------------------------------- #
# Main verification
# --------------------------------------------------------------------------- #
def main(folder=DEFAULT_FOLDER):
    t, x_raw, fs = load_recording(folder)
    x_bp = bandpass_filter(x_raw, fs)
    rms_ref = rms(x_bp)
    fr_ref, mag_ref = rfft_mag(x_bp, fs)
    ref_f0 = peak_near(fr_ref, mag_ref, F0)          # mains spike @ ~50 Hz
    ref_2f0 = peak_near(fr_ref, mag_ref, 2 * F0)     # @ ~100 Hz

    _, M = comb_coeffs(fs, order=1)
    print(f"\nM = round(fs/f0) = {M}   r_m = {R_M}   orders = {list(ORDERS)}")
    print(f"RMS bandpassed (reference) = {rms_ref:.6f} V")
    print(f"Reference mains peaks:  @f0={ref_f0:.5f} V   @2f0={ref_2f0:.5f} V")

    # filtered[(name, version, order)] = y  (kept for the plots)
    filtered = {}

    header = (f"{'filter':<9}{'version':<9}{'order':>6}{'pass|H|':>10}"
              f"{'filtfilt dB':>13}{'att@50':>9}{'att@100':>9}"
              f"{'RMS(V)':>11}{'x ref':>9}")

    for name in ("FIR comb", "IIR comb"):
        print("\n" + "=" * len(header))
        print(name)
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for version, norm in (("ORIGINAL", False), ("PATCHED", True)):
            for order in ORDERS:
                coeffs, _ = comb_coeffs(fs, order=order, normalize=norm)
                b, a = coeffs[name]
                peak = passband_peak(b, a)
                y = sig.filtfilt(b, a, x_bp)
                filtered[(name, version, order)] = y

                fr, mag = rfft_mag(y, fs)
                att_f0 = 20 * np.log10(ref_f0 / (peak_near(fr, mag, F0) + 1e-20))
                att_2f0 = 20 * np.log10(ref_2f0 / (peak_near(fr, mag, 2*F0) + 1e-20))
                r = rms(y)
                print(f"{name:<9}{version:<9}{order:>6}{peak:>10.3f}"
                      f"{20*np.log10(peak**2):>+12.1f}{att_f0:>+9.1f}"
                      f"{att_2f0:>+9.1f}{r:>11.6f}{r/rms_ref:>9.2f}")
            print("-" * len(header))

    print("\nHOW ORDER IMPACTS FILTRATION")
    print("  * Notch rejection (att@50/att@100) gets DEEPER/steeper as order rises.")
    print("  * ORIGINAL passband boost grows as order^ -> 'x ref' explodes")
    print("    (FIR filtfilt peak = 4^order; order 5 => +60 dB, RMS blows up).")
    print("  * PATCHED keeps pass|H| == 1.000 at EVERY order: sharper notches,")
    print("    no amplitude inflation ('x ref' stays <= 1).")
    print("  * Stability holds for all orders (IIR poles keep radius r_m^(1/M)<1).\n")

    # ------------------------------- plots --------------------------------- #
    plot_freq_response_plotly(fs, scale="db")
    plot_freq_response_plotly(fs, scale="linear")
    plot_time_overlay_plotly(t, x_bp, rms_ref, filtered, folder)


def _show_or_note(fig, out):
    """Write an interactive HTML file; also try to open it in a browser."""
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved interactive plot -> {out}")
    try:
        fig.show()          # opens default browser when a display is available
    except Exception:
        pass


def plot_freq_response_plotly(fs, scale="db"):
    """
    filtfilt gain vs frequency: 2x2 = (FIR/IIR) x (ORIGINAL/PATCHED), one line
    per order 1..5.

    scale="db"     -> y = 20*log10(|H|^2): deep notches AND the tall original
                      arches (up to +60 dB) fit on one axis; unity = 0 dB line.
    scale="linear" -> y = |H|^2 (linear magnitude): notches visibly touch 0 and
                      arch heights are true to scale; unity = 1.0 line. ORIGINAL
                      panels auto-range (order 5 FIR reaches ~1024); PATCHED
                      panels are pinned to [0, 1.15] to show the flat unity top.
    """
    linear = (scale == "linear")
    fmax = 3.2 * F0
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("FIR comb - ORIGINAL", "IIR comb - ORIGINAL",
                        "FIR comb - PATCHED",  "IIR comb - PATCHED"),
        horizontal_spacing=0.08, vertical_spacing=0.13)

    layout = [("FIR comb", "ORIGINAL", 1, 1, False),
              ("IIR comb", "ORIGINAL", 1, 2, False),
              ("FIR comb", "PATCHED",  2, 1, True),
              ("IIR comb", "PATCHED",  2, 2, True)]
    unity = 1.0 if linear else 0.0
    y_title = "filtfilt gain |H|^2 (linear)" if linear else "filtfilt gain (dB)"

    for name, version, row, col, norm in layout:
        for order in ORDERS:
            coeffs, _ = comb_coeffs(fs, order=order, normalize=norm)
            b, a = coeffs[name]
            w, H = sig.freqz(b, a, worN=200000, fs=fs)
            m = w <= fmax
            g = np.abs(H[m]) ** 2                              # filtfilt = |H|^2
            y = g if linear else 20 * np.log10(g + 1e-12)
            fig.add_trace(go.Scatter(
                x=w[m], y=y, mode="lines",
                line=dict(color=ORDER_SHADE[name][order - 1], width=1.8),
                name=f"order {order}", legendgroup=f"order{order}",
                showlegend=(row == 1 and col == 1)), row=row, col=col)
        fig.add_hline(y=unity, line=dict(color="grey", dash="dot", width=1),
                      row=row, col=col)
        for k in range(0, int(fmax / F0) + 1):
            fig.add_vline(x=k * F0, line=dict(color="red", dash="dot", width=0.6),
                          opacity=0.35, row=row, col=col)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[0, fmax],
                         row=row, col=col)
        # y-range: dB fixed; linear pins the patched panels, auto-ranges original
        if linear:
            yr = [0, 1.15] if version == "PATCHED" else None
        else:
            yr = [-80, 70]
        fig.update_yaxes(title_text=y_title, range=yr, row=row, col=col)

    scale_note = ("linear magnitude -- notches reach exactly 0; grey line = "
                  "unity (1.0); PATCHED tops sit flat on it at every order"
                  if linear else
                  "0 dB grey line = unity gain; deeper notches AND taller arches "
                  "as order rises (ORIGINAL); PATCHED arches stay at 0 dB")
    fig.update_layout(
        title=(f"Comb filters: filtfilt gain vs frequency, orders 1..5 "
               f"({'linear' if linear else 'dB / log'} scale)"
               f"<br><sup>{scale_note} &nbsp;|&nbsp; red dotted = harmonics of "
               f"f0={F0:.3f} Hz</sup>"),
        template="plotly_white", height=760, hovermode="x unified")
    out = ("comb_filter_freq_response_linear.html" if linear
           else "comb_filter_freq_response.html")
    _show_or_note(fig, out)


def _envelope(t, x, n_bins=4000):
    """
    Peak-preserving decimation for plotting long signals: split into n_bins
    time-bins and keep the ACTUAL min and max sample of each bin. Unlike plain
    striding this never hides a peak, so on-screen amplitude stays truthful
    while the point count drops to ~2*n_bins.
    """
    n = len(x)
    if n <= 2 * n_bins:
        return t, x
    bin_len = n // n_bins
    usable  = bin_len * n_bins
    idx = np.arange(usable).reshape(n_bins, bin_len)
    xb  = x[:usable].reshape(n_bins, bin_len)
    rows = np.arange(n_bins)
    keep = np.union1d(idx[rows, np.argmin(xb, axis=1)],
                      idx[rows, np.argmax(xb, axis=1)])
    if usable < n:                              # keep the leftover tail
        keep = np.union1d(keep, np.arange(usable, n))
    return t[keep], x[keep]


def plot_time_overlay_plotly(t, x_bp, rms_ref, filtered, folder):
    """
    Time-domain amplitude vs order: 2x2 = (FIR/IIR) x (ORIGINAL/PATCHED). Each
    panel draws the band-passed reference (grey) UNDER the filter output at every
    order 1..5 (light->dark). Lets you read off how order changes the amplitude:
    ORIGINAL panels blow up with order, PATCHED panels stay bounded by the ref.
    Axes are shared/linked -- zoom one panel and all four follow.
    """
    layout = [("FIR comb", "ORIGINAL", 1, 1), ("IIR comb", "ORIGINAL", 1, 2),
              ("FIR comb", "PATCHED",  2, 1), ("IIR comb", "PATCHED",  2, 2)]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{n} - {v}" for n, v, _, _ in layout],
        shared_xaxes=True, horizontal_spacing=0.06, vertical_spacing=0.11)

    t_ref, y_ref = _envelope(t, x_bp)
    for name, version, row, col in layout:
        fig.add_trace(go.Scattergl(
            x=t_ref, y=y_ref, mode="lines",
            line=dict(color=C_BANDPASS, width=1),
            name="Bandpassed (ref)", legendgroup="ref",
            showlegend=(row == 1 and col == 1)), row=row, col=col)
        for order in ORDERS:
            t_f, y_f = _envelope(t, filtered[(name, version, order)])
            xr = rms(filtered[(name, version, order)]) / rms_ref
            fig.add_trace(go.Scattergl(
                x=t_f, y=y_f, mode="lines",
                line=dict(color=ORDER_SHADE[name][order - 1], width=1),
                name=f"order {order} (x{xr:.2f})",
                legendgroup=f"{name}{version}o{order}"), row=row, col=col)

    for c in (1, 2):
        fig.update_xaxes(title_text="Time (s)", row=2, col=c)
    for r in (1, 2):
        fig.update_yaxes(title_text="Amplitude (V)", row=r, col=1)
    fig.update_layout(
        title=(f"Time-domain amplitude vs comb order 1..5 &nbsp;|&nbsp; "
               f"{folder}/{CHANNEL}"
               "<br><sup>grey = bandpassed reference (same in every panel); "
               "legend shows RMS x reference. Note ORIGINAL panels autoscale to "
               "their exploding high orders.</sup>"),
        template="plotly_white", height=820, hovermode="x unified")
    _show_or_note(fig, "comb_filter_time_overlay.html")


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
