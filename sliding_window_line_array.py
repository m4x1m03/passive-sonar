# every mic is spaced 6.35 cm apart, track 1 is the rightmost
# (looking from microphone array forward), track 4 the leftmost
# I am taking the example code provided in the library's GitHub to start and modifying
# it to work with my type of files and my specific problems.
# The provided function to plot the data was also throwing errors, so I created my own.

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
from matplotlib.animation import FuncAnimation

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000
nfft = 1024
c = 343.0  # speed of sound
mic_spacing = 0.0635  # 6.35 cm (not good for low-end frequencies)
num_mics = 4
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation (235Hz-2.767kHz range)
chunk_duration = 0.1  # seconds
chunk_size = int(chunk_duration * fs)
hop_size = chunk_size // 2
valid_algorithms = [
    "SRP",
    # "MUSIC",
    # "CSSM",
    # "WAVES",
]

# -----------------------------
# MICROPHONE ARRAY GEOMETRY
# (4 microphones inline along x-axis)
# -----------------------------
R = pra.linear_2D_array([0,0], 4, 0, mic_spacing)

# -----------------------------
# LOAD SIGNALS AND COMBINE IN ONE
# -----------------------------
files = [
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr1.wav",
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr2.wav",
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr3.wav",
    "4mic_array/Raw_Test_Samples/260114_155735/260114_155735_Tr4.wav"
]

signals = []
sample_rates = []

for f in files:
    fs, audio = wavfile.read(f)
    sample_rates.append(fs)

    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

        # Normalize integer audio
        if audio.max() > 1.0:
            audio /= np.iinfo(np.int16).max

    signals.append(audio)

signals = np.stack(signals, axis=0)


# -----------------------------
# Compute the STFT frames needed
# -----------------------------
num_samples = signals.shape[1]
num_chunks = (num_samples - chunk_size) // hop_size + 1
times = []
azimuths = []

doa = pra.doa.algorithms[valid_algorithms](R, fs, nfft, c=c, max_four=4)

for i in range(num_chunks):
    start = i * hop_size
    end = start + chunk_size

    chunk = signals[:, start:end]

    # STFT for this chunk
    X_chunk = np.array(
        [
            pra.transform.stft.analysis(sig, nfft, nfft // 2).T
            for sig in chunk
        ]
    )

    doa.locate_sources(X_chunk, freq_bins=freq_bins)

    if doa.azimuth_recon.size > 0:
        az = doa.azimuth_recon[0]
    else:
        az = np.nan

    azimuths.append(az)
    times.append(start / fs)


# -----------------------------
# Plot DOA from SRP and animated
# -----------------------------
azimuths_deg = np.array(azimuths) * 180 / np.pi
azimuths = np.array(azimuths)
times = np.array(times)

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# Initial DOA line
(doa_line,) = ax.plot([0, 0], [0, 1.0], "r-", linewidth=2)

# Time text
time_text = ax.text(
    0.02, 0.95, "", transform=ax.transAxes, color="black"
)

ax.set_rmax(1.0)
ax.set_rticks([])
ax.set_theta_zero_location("E")   # 0° = +x
ax.set_theta_direction(1)         # CCW
ax.set_title("DOA Tracking (0.1 s chunks)")

def update(frame):
    az = azimuths[frame]

    if np.isnan(az):
        doa_line.set_data([], [])
    else:
        doa_line.set_data([az, az], [0, 1.0])

    time_text.set_text(f"t = {times[frame]:.2f} s")

    return doa_line, time_text

interval_ms = int(chunk_duration * 1000 * hop_size / chunk_size)

ani = FuncAnimation(
    fig,
    update,
    frames=len(azimuths),
    interval=interval_ms,
    blit=True,
    repeat=False,
)

plt.show()
