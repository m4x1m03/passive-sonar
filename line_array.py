# every mic is spaced 6.35 cm apart, track 1 is the rightmost
# (looking from microphone array forward), track 4 the leftmost
# I am taking the example code provided in the library's GitHub to start and modifying
# it to work with my type of files and my specific problems.
# The provided function to plot the data was also throwing errors, so I created my own.

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 48000
nfft = 1024
c = 343.0  # speed of sound
mic_spacing = 0.0635  # 6.35 cm (not good for low-end frequencies)
num_mics = 4
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation (235Hz-2.767kHz range)
valid_algorithms = [
    "SRP",
    "MUSIC",
    "CSSM",
    "WAVES",
]
# chunk_duration = 0.1  # seconds
# chunk_samples = int(chunk_duration * fs)
# hop_samples = chunk_samples // 2  # 50% overlap


# -----------------------------
# MICROPHONE ARRAY GEOMETRY
# (4 microphones inline along x-axis)
# -----------------------------
R = pra.linear_2D_array([0,0], 4, 0, mic_spacing)


# -----------------------------
# LOAD SIGNALS AND COMBINE IN ONE
# -----------------------------
files = [
    "4mic_array/Test_Side1/Right_Tr1.wav",
    "4mic_array/Test_Side1/Right_Tr2.wav",
    "4mic_array/Test_Side1/Right_Tr3.wav",
    "4mic_array/Test_Side1/Right_Tr4.wav"
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
# Break down the signals in chunks
# -----------------------------
# def chunk_signals(signals, chunk_samples, hop_samples):
#     num_mics, num_samples = signals.shape
#     for start in range(0, num_samples - chunk_samples, hop_samples):
#         yield signals[:, start : start + chunk_samples]
#

# -----------------------------
# PLOTTING
# -----------------------------
def plot_doa_polar(azimuth_est, azimuth_true=None, title="DOA"):
    """
    azimuth_est : scalar or array-like (radians)
    azimuth_true: scalar (radians) or None
    """
    azimuth_est = np.atleast_1d(azimuth_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Estimated DOAs (red)
    for az in azimuth_est:
        ax.plot([az, az], [0, 1.0], "r-", linewidth=2)

    # True DOA (green)
    if azimuth_true is not None:
        ax.plot([azimuth_true, azimuth_true], [0, 1.0], "g--", linewidth=2)

    ax.set_rmax(1.0)
    ax.set_rticks([])
    ax.set_theta_zero_location("E")   # 0° = +x axis
    ax.set_theta_direction(1)          # CCW positive
    ax.set_title(title)

    return ax

# -----------------------------
# Compute the STFT frames needed
# -----------------------------
# direction_history = []
#
# for chunk in chunk_signals(signals, chunk_samples, hop_samples):
#         X = np.array(
#         [
#             pra.transform.stft.analysis(signal, nfft, nfft // 2).T
#             for signal in signals
#         ]
#         )
#
#         for algo_name in valid_algorithms:
#             doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c)
#             doa.locate_sources(X, freq_bins=freq_bins)
#
#             print(algo_name)
#             print(
#                 "  Recovered azimuth:",
#                 np.atleast_1d(doa.azimuth_recon) / np.pi * 180.0,
#                 "degrees",
#             )
#             direction_history.append(doa.azimuth_recon)
#
#
# print(direction_history)
# print(len(direction_history))

################################
# Compute the STFT frames needed
X = np.array(
    [
        pra.transform.stft.analysis(signal, nfft, nfft // 2).T
        for signal in signals
    ]
)


# PLOTTING
def plot_doa_polar(azimuth_est, azimuth_true=None, title="DOA"):
    """
    azimuth_est : scalar or array-like (radians)
    azimuth_true: scalar (radians) or None
    """
    azimuth_est = np.atleast_1d(azimuth_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Estimated DOAs (red)
    for az in azimuth_est:
        ax.plot([az, az], [0, 1.0], "r-", linewidth=2)

    # True DOA (green)
    if azimuth_true is not None:
        ax.plot([azimuth_true, azimuth_true], [0, 1.0], "g--", linewidth=2)

    ax.set_rmax(1.0)
    ax.set_rticks([])
    ax.set_theta_zero_location("E")   # 0° = +x axis
    ax.set_theta_direction(1)          # CCW positive
    ax.set_title(title)

    return ax

def mirrored_azimuth(az):
    az = np.atleast_1d(az)
    return np.concatenate([az, -az])


##############################################
# Now we can test all the algorithms available
algo_names = sorted(pra.doa.algorithms.keys())

for algo_name in valid_algorithms:
    doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, max_four=4)
    doa.locate_sources(X, freq_bins=freq_bins)

    # Plot without pyroomacoustics
    plot_doa_polar(
        azimuth_est=mirrored_azimuth(doa.azimuth_recon),
        title=f"{algo_name} (front/back)",
    )

    print(algo_name)
    print(
        "  Recovered azimuth:",
        np.atleast_1d(doa.azimuth_recon) / np.pi * 180.0,
        "degrees",
    )


plt.show()