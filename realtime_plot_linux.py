"""
Real-time EMG plot from Axagon ADA-71 on Arch Linux.

Streams 2 channels from the ALSA device (ICUSBAUDIO7D: USB Audio)
and displays a scrolling waveform plot for both channels.

Press Ctrl+C or close the window to stop.
"""

import threading
import collections
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── settings ─────────────────────────────────────────────────────────────────
DEVICE      = "*ICUSBAUDIO7D*"   # matches "ICUSBAUDIO7D: USB Audio (hw:4,0)"
HOSTAPI     = "ALSA"
CHANNELS    = 2
BLOCK_SIZE  = 512
WINDOW_SEC  = 3.0      # seconds of signal visible in the plot
YLIM        = (-0.5, 0.5)   # V — adjust to your signal amplitude
INTERVAL_MS = 40            # plot refresh interval


# ── device resolution ────────────────────────────────────────────────────────

def _find_device(pattern: str, channels: int) -> tuple[int, int]:
    from fnmatch import fnmatch
    devices  = sd.query_devices()
    hostapis = sd.query_hostapis()
    pat = pattern.lower()
    for i, dev in enumerate(devices):
        api = hostapis[dev["hostapi"]]["name"]
        if (fnmatch(dev["name"].lower(), pat)
                and HOSTAPI.lower() in api.lower()
                and dev["max_input_channels"] >= channels):
            return i, int(dev["default_samplerate"])
    raise ValueError(
        f"No {HOSTAPI} device matching {pattern!r} with >= {channels} ch.\n"
        f"Run: python -c \"import sounddevice; print(sounddevice.query_devices())\""
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dev_id, sample_rate = _find_device(DEVICE, CHANNELS)
    dev_name = sd.query_devices(dev_id)["name"].strip()
    print(f"Device [{dev_id}]: {dev_name} @ {sample_rate} Hz, {CHANNELS} ch")

    window_samples = int(WINDOW_SEC * sample_rate)
    buffers = [collections.deque(np.zeros(window_samples), maxlen=window_samples)
               for _ in range(CHANNELS)]
    lock = threading.Lock()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream] {status}")
        with lock:
            for ch in range(CHANNELS):
                buffers[ch].extend(indata[:, ch])

    # ── plot setup ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(CHANNELS, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(f"EMG — {dev_name}")
    ch_labels = ["Line In L", "Line In R"]
    colors     = ["steelblue", "tomato"]
    lines = []

    x = np.arange(window_samples) / sample_rate

    for ch, ax in enumerate(axes):
        (line,) = ax.plot(x, np.zeros(window_samples), lw=0.8, color=colors[ch])
        lines.append(line)
        ax.set_ylim(*YLIM)
        ax.set_ylabel(f"{ch_labels[ch]} (V)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    def update(_frame):
        with lock:
            snapshots = [np.array(buf) for buf in buffers]
        for ch, line in enumerate(lines):
            line.set_ydata(snapshots[ch])
        return lines

    with sd.InputStream(device=dev_id,
                        samplerate=sample_rate,
                        channels=CHANNELS,
                        blocksize=BLOCK_SIZE,
                        callback=callback):
        print("Streaming — close the plot window or press Ctrl+C to stop.")
        _anim = FuncAnimation(fig, update, interval=INTERVAL_MS, blit=True)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    print("Stopped.")


if __name__ == "__main__":
    main()
