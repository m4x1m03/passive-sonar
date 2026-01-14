import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

######
# We define a meaningful distance measure on the circle

# Location of original source
azimuth = 170.0 / 180.0 * np.pi  # 60 degrees
distance = 3.0  # 3 meters
dim = 2  # dimensions (2 or 3)
room_dim = np.r_[10.0, 10.0]
valid_algorithms = [
    "SRP",
    "MUSIC",
    "CSSM",
    "WAVES",
]

# Use AnechoicRoom or ShoeBox implementation. The results are equivalent because max_order=0 for both.
# The plots change a little because in one case there are no walls.
use_anechoic_class = True


#######################
# algorithms parameters
SNR = 0.0  # signal-to-noise ratio
c = 343.0  # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# compute the noise variance
sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2

# Create an anechoic room
if use_anechoic_class:
    aroom = pra.AnechoicRoom(dim, fs=fs, sigma2_awgn=sigma2)
else:
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

# add the source
source_location = room_dim / 2 + distance * np.r_[np.cos(azimuth), np.sin(azimuth)]
source_signal = np.random.randn((nfft // 2 + 1) * nfft)
aroom.add_source(source_location, signal=source_signal)

# We use a linear array of 4 microphones
R = pra.linear_2D_array(room_dim/2, 4, 0, 0.0635)
aroom.add_microphone_array(pra.MicrophoneArray(R, fs=aroom.fs))

# run the simulation
aroom.simulate()

################################
# Compute the STFT frames needed
X = np.array(
    [
        pra.transform.stft.analysis(signal, nfft, nfft // 2).T
        for signal in aroom.mic_array.signals
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
        azimuth_true=azimuth,
        title=f"{algo_name} (front/back)",
    )

    print(algo_name)
    print(
        "  Recovered azimuth:",
        np.atleast_1d(doa.azimuth_recon) / np.pi * 180.0,
        "degrees",
    )
    print(
        "  Error:",
        circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180.0,
        "degrees",
    )

plt.show()
