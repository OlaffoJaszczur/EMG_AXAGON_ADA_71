# EMG Data Acquisition from Axagon ADA-71 Sound Card

Python tools for acquiring, visualizing, and filtering EMG (electromyography) signals from the Axagon ADA-71 USB audio interface — including powerline-removal filters (notch, comb, adaptive LMS) and standalone studies of filter order and passband normalization.

## Setup

### 1. Install Dependencies

**Windows:**
```bash
pip install -r requirements.txt
```

**Arch Linux:**
```bash
sudo pacman -S portaudio python-sounddevice python-numpy
pip install -r requirements.txt
```

### 2. Check Your Device
```bash
python -m sounddevice
```

- **Windows:** The ADA-71 shows up as `"USB Sound Device"` with separate Line In and Mic In endpoints. Note that its **Microphone** jack is mono (signal on the left/tip contact only) — for two-channel recording, use the stereo **Line In** jack.
- **Linux (ALSA):** The ADA-71 (CM106 chip) exposes a single PCM device: `"ICUSBAUDIO7D: USB Audio"` with 2 input channels. Both Line In and Mic In share this device; the active capture source is toggled in `alsamixer`.

**Hardware limitation:** The ADA-71 only allows one capture endpoint active at a time. Simultaneous Line In + Mic In is not supported; `dual_acquisition.py` / `dual_acquisition_linux.py` records them sequentially.

### 3. Configure the Device

**Windows** — acquisition device selection lives at the top of [emg_acquisition.py](emg_acquisition.py):

- `ACQUISITION_DEVICE`: a sounddevice index (int) or a name pattern (str, case-insensitive, `*` wildcards allowed). Default: `"Line*USB Sound Device*"`.
- `ACQUISITION_HOSTAPI`: preferred host API substring (e.g. `"WASAPI"`, `"DirectSound"`, `"MME"`). WASAPI has the lowest latency.

**Linux** — device settings are at the top of [dual_acquisition_linux.py](dual_acquisition_linux.py) and [realtime_plot_linux.py](realtime_plot_linux.py):

- `LINE_DEVICE` / `DEVICE`: ALSA device name pattern. Default: `"*ICUSBAUDIO7D*"`.
- `HOSTAPI`: host API to match. Default: `"ALSA"`.

To switch the active capture source between Line In and Mic, run:
```bash
alsamixer -c <card_number>   # press Space to toggle capture on a source
```

---

## Files

| File | Platform | Description |
|------|----------|-------------|
| [emg_acquisition.py](emg_acquisition.py) | Windows | Core recording class — NPZ + JSON output |
| [dual_acquisition.py](dual_acquisition.py) | Windows | Sequential Line In + Mic recording — CSV output |
| [dual_acquisition_linux.py](dual_acquisition_linux.py) | Linux (ALSA) | Sequential Line In + Mic recording — CSV output |
| [realtime_plot.py](realtime_plot.py) | Windows | Live oscilloscope view while recording |
| [realtime_plot_linux.py](realtime_plot_linux.py) | Linux (ALSA) | Live oscilloscope view while recording |
| [quickstart.py](quickstart.py) | Windows | 5-second test recording with stats and plot |
| [emg_analysis.ipynb](emg_analysis.ipynb) | Any | Notebook: bandpass + comb/notch filter analysis |
| [emg_lms.ipynb](emg_lms.ipynb) | Any | Notebook: Block NLMS adaptive powerline canceller |
| [filter_comparison.py](filter_comparison.py) | Any | Script: FIR vs IIR × Notch vs Comb filter comparison |
| [filter_comparison_notebook.ipynb](filter_comparison_notebook.ipynb) | Any | Notebook: interactive version of filter_comparison.py |
| [comb_filter_verification.py](comb_filter_verification.py) | Any | Script: comb-filter order (1–10) sweep + passband-normalization study → interactive Plotly HTML |
| [notch_filter_verification.py](notch_filter_verification.py) | Any | Script: notch-filter order (1–10) sweep → interactive Plotly HTML |
| [notch_filter_normalized.py](notch_filter_normalized.py) | Any | Script: notch filters normalized to unity passband (original vs normalized) → interactive Plotly HTML |

---

## Usage

### Quick Start
```bash
python quickstart.py
```
Runs a 5-second test recording, prints statistics, saves the data, and plots it.

### Dual-Channel Acquisition (Line In + Mic)

**Windows:**
```bash
python dual_acquisition.py
```

**Linux (Arch):**
```bash
python dual_acquisition_linux.py
```

Records Line In (stereo, 2 ch) and Microphone (mono, 1 ch) in sequence, saving each channel as a timestamped CSV in `recordings/emg_rec_<timestamp>/`:

| File | Contents |
|------|----------|
| `emg_line_L.csv` | Line In — left channel |
| `emg_line_R.csv` | Line In — right channel |
| `emg_mic.csv` | Microphone (mono) |

Each CSV has two columns: `time [s]` and `data [V]`.

On Linux, both recordings use the same ALSA device (`ICUSBAUDIO7D: USB Audio`). Switch the active capture source between the two runs using `alsamixer`.

### Real-Time Visualization

**Windows:**
```bash
python realtime_plot.py
```

**Linux (Arch):**
```bash
python realtime_plot_linux.py
```

Shows a scrolling 2-channel waveform of Line In (left and right) while recording. Close the window or press Ctrl+C to stop.

### Signal Analysis (Jupyter)

Open [emg_analysis.ipynb](emg_analysis.ipynb) and run all cells. The interactive widget lets you:

- Select a recording folder from `recordings/`
- Choose between **Feedforward Comb** filter (`y[n] = x[n] − x[n−M]`, nulls at 50, 100, 150 … Hz) or **IIR Notch** filter (cascade of zero-phase 2nd-order notches at each harmonic)
- Set FFT display range

**Output per channel — 2 Plotly figures:**
- Fig A: 3 rows (Raw / Bandpass / Filtered) × 2 cols (time domain | FFT)
- Fig B: FFT overlay of all three stages with 50 Hz harmonic markers

### Filter Comparison

**Script:**
```bash
python filter_comparison.py                        # uses default recording
python filter_comparison.py emg_rec_20260625_092258
```

**Notebook:** open [filter_comparison_notebook.ipynb](filter_comparison_notebook.ipynb) and run all cells.

Loads `emg_line_L.csv` from a recordings subfolder, applies a 6–500 Hz bandpass, then benchmarks four powerline-removal strategies side by side:

| Filter | Type | Method |
|--------|------|--------|
| FIR Notch | FIR | `firwin` bandstop cascade (Hamming window) |
| IIR Notch | IIR | `iirnotch` biquad cascade (Q=30) |
| FIR Comb | FIR | Feedforward comb: `y[n] = x[n] − x[n−M]` |
| IIR Comb | IIR | Feedback comb: `y[n] = x[n] − x[n−M] + rᴹ·y[n−M]` |

All harmonics of F₀ up to 500 Hz are cancelled. Output figures show time-domain waveforms, FFT overlays, and a performance summary table (SNR, computation time).

> **Note:** FIR notch requires impractically many taps at audio sample rates (≫ 10 000 at fs=44100, bw=5 Hz). The script caps `MAX_FIR_TAPS=4001` intentionally to illustrate this limitation — the resulting notch is wider than ideal.

### Filter Order & Normalization Studies

Three standalone scripts drill deeper into a single filter family, sweeping the
**cascade order 1–10** and measuring how order trades off harmonic rejection,
EMG preservation, passband gain, and compute time. Each loads `emg_line_L.csv`
from a recording, applies the 6–500 Hz bandpass reference, then writes a set of
**self-contained interactive Plotly HTML files** (open in any browser — drag to
zoom, click the legend to toggle traces) plus a color-coded console table.

```bash
python comb_filter_verification.py [recording_folder]   # comb filters
python notch_filter_verification.py [recording_folder]  # notch filters
python notch_filter_normalized.py [recording_folder]    # notch normalization
```

| Script | Focus | HTML outputs |
|--------|-------|--------------|
| [comb_filter_verification.py](comb_filter_verification.py) | FIR/IIR **comb**, order 1–10, **ORIGINAL vs PATCHED** (passband normalized). Shows the comb's between-harmonic passband arch (`4^order` for FIR under `filtfilt`, up to +60 dB), the normalization that fixes it, and per-order timing. | `comb_filter_results_table`, `comb_filter_freq_response` (dB), `comb_filter_freq_response_linear`, `comb_filter_time_overlay` |
| [notch_filter_verification.py](notch_filter_verification.py) | FIR/IIR **notch**, order 1–10, with a recommended order. IIR notch (biquad SOS cascade) preserves EMG and is cheap; the capped FIR notch is wide and destroys EMG. | `notch_filter_results_table`, `notch_filter_freq_response` (dB), `notch_filter_freq_response_linear`, `notch_filter_time_overlay` |
| [notch_filter_normalized.py](notch_filter_normalized.py) | Notch **passband normalized to 1** (ORIGINAL vs NORMALIZED). Confirms the IIR notch is already exactly unity (normalization is a no-op) while the FIR notch has only a small window-ripple overshoot (≤ +0.43 dB at order 10). | `notch_norm_results_table`, `notch_norm_freq_response` (dB), `notch_norm_freq_response_linear` |

Every results table reports, per filter × order, the passband peak, filtfilt
gain (dB **and** linear), harmonic attenuation at 50/100 Hz (dB **and** linear),
RMS vs the bandpass reference, and the apply time.

**Key findings from these studies**

- **Comb passband amplification.** The feedforward comb `1 − z⁻ᴹ` arches to a gain
  of 2 between harmonics; `filtfilt` squares that to ×4 (+12 dB), and higher order
  compounds it (`4^order`). The IIR comb `(1 − z⁻ᴹ)/(1 − r_m·z⁻ᴹ)` arches less but
  still above 1. Scaling the numerator (FIR ×½, IIR ×(1+r_m)/2 per section, or the
  whole cascade to unity) removes the boost. **Recommended: FIR comb order 1, IIR
  comb order ~6.**
- **Notch needs (almost) no normalization.** Each notch section has passband gain
  ≤ 1 by construction, so — unlike the comb — the notch never inflates the signal.
  The IIR biquad notch is *exactly* unity; the FIR notch overshoots by at most a
  fraction of a dB (Hamming ripple). **Recommended: IIR notch order ~6** (tuned to
  the exact mains, preserves EMG, ~30 ms); the FIR notch is unsuitable at any order
  because its capped-tap notch is ~88 Hz wide and removes real EMG.
- **High-order IIR must use second-order sections.** Expanding a cascaded biquad
  into one degree-2N polynomial explodes numerically (poles pile up on the unit
  circle); `notch_filter_verification.py` uses `sosfiltfilt` to stay stable.

### LMS Adaptive Filtering (Jupyter)

Open [emg_lms.ipynb](emg_lms.ipynb) and run all cells. Uses a **Block NLMS** adaptive filter to estimate and subtract powerline interference.

**How it works:** A synthetic reference is built from `sin + cos` at `f0, 2f0, … n_harmonics × f0`. The filter finds weights such that `ref @ w ≈ interference`, then returns `e = signal − ref @ w`. Weights update once per block (fully vectorised). Small mains frequency drift is tracked automatically.

**Adjustable parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Reference freq f₀ | 50.0 Hz | Mains frequency (slider: 49–51 Hz) |
| Step size μ | 0.1 | NLMS convergence rate |
| Harmonics | 5 | Number of harmonics cancelled |
| Block size | 256 | Samples per weight update |

**Output per channel — 2 Plotly figures:**
- Fig A: 3 rows (Raw / Bandpass / LMS) × 2 cols (time domain | FFT)
- Fig B: FFT overlay of all three stages with harmonic markers

---

## Signal Processing Pipeline

```
Raw signal
  └─ Bandpass filter (6–500 Hz, 4th-order Butterworth, zero-phase SOS)
       ├─ FIR Notch      (firwin bandstop cascade)           [filter_comparison.py / notch_filter_verification.py]
       ├─ IIR Notch      (iirnotch biquad cascade, Q=30)     [emg_analysis.ipynb / notch_filter_verification.py]
       ├─ FIR Comb       (y[n] = x[n] − x[n−M])             [emg_analysis.ipynb / comb_filter_verification.py]
       ├─ IIR Comb       (y[n] = x[n] − x[n−M] + rᴹ·y[n−M])[comb_filter_verification.py]
       └─ Block NLMS     (adaptive interference cancel)       [emg_lms.ipynb]
```

The `*_verification.py` scripts additionally sweep the **cascade order (1–10)** of
a filter family and study **passband normalization** — see
[Filter Order & Normalization Studies](#filter-order--normalization-studies).

---

## Configuration

### Sample Rate
- `sample_rate=None` (default): Uses the device's default — 48000 Hz for the ADA-71 under WASAPI (Windows) and ALSA (Linux)
- **44100 Hz**: Standard audio
- **48000 Hz**: Professional audio standard

### EMG Signal Considerations
- Typical EMG amplitude: 10 µV – 5 mV
- Useful bandwidth: 10 Hz – 1 kHz
- May need amplification/conditioning before sound card input

---

## File Outputs

| Source | Files |
|--------|-------|
| `dual_acquisition.py` | `recordings/emg_rec_<ts>/emg_line_L.csv`, `emg_line_R.csv`, `emg_mic.csv` |
| `quickstart.py` / `emg_acquisition.py` | `emg_data_<ts>.npz` + `emg_data_<ts>_metadata.json` |

---

## Troubleshooting

### Windows
- **Device not found**: Check `sd.query_devices()`, verify USB connection
- **No signal**: Check microphone/line input levels in Windows Sound settings
- **Buffer underrun**: Reduce `BLOCK_SIZE` or close other audio apps
- **Only one device records**: ADA-71 hardware limitation — Line In and Mic In cannot stream simultaneously

### Linux (Arch)
- **Device not found**: Run `python -c "import sounddevice; print(sounddevice.query_devices())"` and confirm `ICUSBAUDIO7D` appears; try replugging the USB device
- **No signal / wrong source**: Open `alsamixer -c <card_number>`, navigate to the capture section (F4), and toggle the desired source with Space
- **`portaudio` missing**: Install with `sudo pacman -S portaudio`
- **Permission denied on `/dev/snd/*`**: Add yourself to the `audio` group — `sudo usermod -aG audio $USER` — then log out and back in

### General
- **Noise**: Ensure proper grounding of EMG sensors and cables
- **Buffer underrun**: Reduce `BLOCK_SIZE` or close other audio applications
