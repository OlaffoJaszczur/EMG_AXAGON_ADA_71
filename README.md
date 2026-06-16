# EMG Data Acquisition from Axagon ADA-71 Sound Card

Python tools for acquiring, visualizing, and filtering EMG (electromyography) signals from the Axagon ADA-71 USB audio interface.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Your Device
```bash
python -m sounddevice
```
Ensure you see your Axagon ADA-71 (shows up in Windows as "USB Sound Device") with at least 2 input channels. Note that its **Microphone** jack is mono (signal on the left/tip contact only) — for two-channel recording, use the stereo **Line In** jack.

**Hardware limitation:** The ADA-71 (C-Media USB chip) only allows one capture endpoint active at a time. Simultaneous Line In + Mic In is not supported; `dual_acquisition.py` records them sequentially.

### 3. Configure the Device
Acquisition device selection lives at the top of [emg_acquisition.py](emg_acquisition.py):

- `ACQUISITION_DEVICE`: a sounddevice index (int) or a name pattern (str, case-insensitive, `*` wildcards allowed). Default: `"Line*USB Sound Device*"`.
- `ACQUISITION_HOSTAPI`: preferred host API substring (e.g. `"WASAPI"`, `"DirectSound"`, `"MME"`). WASAPI has the lowest latency.

---

## Files

| File | Description |
|------|-------------|
| [emg_acquisition.py](emg_acquisition.py) | Core recording class — NPZ + JSON output |
| [dual_acquisition.py](dual_acquisition.py) | Sequential Line In + Mic recording — CSV output |
| [realtime_plot.py](realtime_plot.py) | Live oscilloscope view while recording |
| [quickstart.py](quickstart.py) | 5-second test recording with stats and plot |
| [emg_analysis.ipynb](emg_analysis.ipynb) | Notebook: bandpass + comb/notch filter analysis |
| [emg_lms.ipynb](emg_lms.ipynb) | Notebook: Block NLMS adaptive powerline canceller |

---

## Usage

### Quick Start
```bash
python quickstart.py
```
Runs a 5-second test recording, prints statistics, saves the data, and plots it.

### Dual-Channel Acquisition (Line In + Mic)
```bash
python dual_acquisition.py
```
Records Line In (stereo, 2 ch) and Microphone (mono, 1 ch) in sequence, saving each channel as a timestamped CSV in `recordings/emg_rec_<timestamp>/`:

| File | Contents |
|------|----------|
| `emg_line_L.csv` | Line In — left channel |
| `emg_line_R.csv` | Line In — right channel |
| `emg_mic.csv` | Microphone (mono) |

Each CSV has two columns: `time [s]` and `data [V]`.

### Real-Time Visualization
```bash
python realtime_plot.py
```
Shows live EMG signals from both channels while recording.

### Signal Analysis (Jupyter)

Open [emg_analysis.ipynb](emg_analysis.ipynb) and run all cells. The interactive widget lets you:

- Select a recording folder from `recordings/`
- Choose between **Feedforward Comb** filter (`y[n] = x[n] − x[n−M]`, nulls at 50, 100, 150 … Hz) or **IIR Notch** filter (cascade of zero-phase 2nd-order notches at each harmonic)
- Set FFT display range

**Output per channel — 2 Plotly figures:**
- Fig A: 3 rows (Raw / Bandpass / Filtered) × 2 cols (time domain | FFT)
- Fig B: FFT overlay of all three stages with 50 Hz harmonic markers

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
       ├─ Comb filter    (y[n] = x[n] − x[n−M])        [emg_analysis.ipynb]
       ├─ IIR Notch      (cascade at 50, 100, … Hz)     [emg_analysis.ipynb]
       └─ Block NLMS     (adaptive interference cancel)  [emg_lms.ipynb]
```

---

## Configuration

### Sample Rate
- `sample_rate=None` (default): Uses the device's default (48000 Hz for the ADA-71 under WASAPI)
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

- **Device not found**: Check `sd.query_devices()`, verify USB connection
- **No signal**: Check microphone/line input levels in Windows Sound settings
- **Noise**: Ensure proper grounding of EMG sensors and cables
- **Buffer underrun**: Reduce `BLOCK_SIZE` or close other audio apps
- **Only one device records**: ADA-71 hardware limitation — Line In and Mic In cannot stream simultaneously
