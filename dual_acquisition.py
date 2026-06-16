"""
Record from Line In (stereo, 2 ch) and front Microphone (mono, 1 ch),
save each channel as its own CSV: time [s], data [V].

Output files (timestamped):
  emg_line_<ts>_L.csv   -- Line In left channel
  emg_line_<ts>_R.csv   -- Line In right channel
  emg_mic_<ts>.csv      -- Microphone (mono)

HARDWARE NOTE
-------------
The Axagon ADA-71 (C-Media USB chip) only allows one capture endpoint to be
active at a time. Simultaneous recording from both Line In and Mic In is NOT
supported by the hardware/driver. This script will record each input in
sequence: first Line In for DURATION seconds, then Mic In for DURATION seconds.
"""

import sys
import ctypes
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

# ── settings ─────────────────────────────────────────────────────────────────
LINE_DEVICE = "Line*USB Sound Device*"
MIC_DEVICE  = "Microphone*USB Sound Device*"
HOSTAPI     = "WASAPI"

DURATION    = 5        # seconds to record per input
BLOCK_SIZE  = 2048
OUTPUT_DIR  = Path(__file__).parent / "recordings"


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_device(pattern: str, channels: int) -> tuple[int, int]:
    """Return (device_index, sample_rate) for the first WASAPI device matching pattern."""
    from fnmatch import fnmatch
    devices  = sd.query_devices()
    hostapis = sd.query_hostapis()
    pat = f"*{pattern.lower()}*"
    for i, dev in enumerate(devices):
        api = hostapis[dev["hostapi"]]["name"]
        if (fnmatch(dev["name"].lower(), pat)
                and HOSTAPI.lower() in api.lower()
                and dev["max_input_channels"] >= channels):
            return i, int(dev["default_samplerate"])
    raise ValueError(
        f"No WASAPI device found matching {pattern!r} with >= {channels} input channel(s)."
    )


def _save_csv(data_1d: np.ndarray, sample_rate: int, filepath: str) -> None:
    time_axis = np.arange(len(data_1d)) / sample_rate
    np.savetxt(filepath,
               np.column_stack((time_axis, data_1d)),
               delimiter=',', header='time,data', comments='', fmt='%.10g')
    print(f"  Saved: {filepath}  ({len(data_1d) / sample_rate:.2f} s, {len(data_1d)} samples)")


def _record(device_id: int, sample_rate: int, channels: int, duration: float) -> np.ndarray | None:
    """Record from a device for `duration` seconds, return (n_samples, channels) array."""
    buffer: list[np.ndarray] = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"    [stream] {status}")
        with lock:
            buffer.append(indata.copy())

    def run():
        # WASAPI is COM-based; COM must be initialised on the thread that opens the stream.
        com_init = False
        if sys.platform == "win32":
            hr = ctypes.windll.ole32.CoInitializeEx(None, 0)
            com_init = hr in (0, 1)
        try:
            with sd.InputStream(device=device_id,
                                samplerate=sample_rate,
                                channels=channels,
                                blocksize=BLOCK_SIZE,
                                callback=callback):
                stop_event.wait()
        except Exception as e:
            print(f"    Stream error: {e}")
        finally:
            if com_init:
                ctypes.windll.ole32.CoUninitialize()

    t = threading.Thread(target=run, daemon=True)
    t.start()

    import time
    time.sleep(duration)
    stop_event.set()
    t.join(timeout=3)

    with lock:
        if not buffer:
            return None
        return np.vstack(buffer)


# ── main ─────────────────────────────────────────────────────────────────────

def main(duration: float = DURATION):
    print("Resolving devices...")
    line_dev, line_sr = _resolve_device(LINE_DEVICE, channels=2)
    mic_dev,  mic_sr  = _resolve_device(MIC_DEVICE,  channels=1)
    print(f"  Line In [{line_dev}]: {sd.query_devices(line_dev)['name'].strip()} @ {line_sr} Hz (2 ch)")
    print(f"  Mic In  [{mic_dev}]:  {sd.query_devices(mic_dev)['name'].strip()} @ {mic_sr} Hz (1 ch)")

    print("\n-- Hardware note: ADA-71 supports only one capture endpoint at a time.")
    print("-- Recording sequentially: Line In first, then Mic.\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"emg_rec_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {session_dir}\n")

    # --- Line In ---------------------------------------------------------------
    print(f"[1/2] Recording Line In for {duration} s...")
    line_data = _record(line_dev, line_sr, channels=2, duration=duration)
    if line_data is not None:
        _save_csv(line_data[:, 0], line_sr, session_dir / "emg_line_L.csv")
        _save_csv(line_data[:, 1], line_sr, session_dir / "emg_line_R.csv")
    else:
        print("  Line In: no data captured.")

    # --- Mic In ----------------------------------------------------------------
    print(f"\n[2/2] Recording Mic In for {duration} s...")
    mic_data = _record(mic_dev, mic_sr, channels=1, duration=duration)
    if mic_data is not None:
        _save_csv(mic_data[:, 0], mic_sr, session_dir / "emg_mic.csv")
    else:
        print("  Mic: no data captured.")

    print("\nDone.")


if __name__ == "__main__":
    main()
