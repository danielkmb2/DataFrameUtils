import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def blackman_filter(series: pd.Series, window_len=41) -> pd.Series:

    # black magic
    def blackman(m):
        if m < 1:
            return np.array([])
        if m == 1:
            return np.ones(1, float)
        n = np.arange(0, m)
        return 0.42 - 0.5 * np.cos(2.0 * np.pi * n / (m - 1)) + 0.08 * np.cos(4.0 * np.pi * n / (m - 1))

    x = series.values
    x = x[int(window_len / 2):-int(window_len / 2)]

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = blackman(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')

    return pd.Series(y)


def savitzky_goly_filter(series: pd.Series, window_len=41) -> pd.Series:
    filtered_series = savgol_filter(series.values, window_len, 5)  # window size 51, polynomial order 3
    return pd.Series(filtered_series)


def median_rolling_filter(series: pd.Series, window_len=41) -> pd.Series:
    return series.rolling(window=window_len, center=True).median().fillna(0)


if __name__ == '__main__':
    def normalise(x):
        maxamp = max(x)
        amp = np.floor(10000 / maxamp)
        norm = np.zeros(len(x))
        for _ in range(len(x)):
            norm[_] = amp * x[_]
        return norm


    def gen_sine(f0, fs, dur):
        t = np.arange(dur)
        sinusoid = np.sin(2 * np.pi * t * (f0 / fs))
        sinusoid = normalise(sinusoid)
        return sinusoid


    def gen_noise(dur):
        _noise = np.random.normal(0, 1, dur)
        _noise = normalise(_noise)
        return _noise


    sin = pd.Series(gen_sine(1, 100, 500))
    noise = gen_noise(500)
    noised_sin = sin + noise
    noised_sin = pd.Series(noised_sin)

    _df = pd.DataFrame({
        "sin": sin,
        "noised_sin": noised_sin,
        "filtered_savgol_sin": savitzky_goly_filter(noised_sin),
        "filtered_blackman_sin": blackman_filter(noised_sin),
        "filtered_median_rolling_sin": median_rolling_filter(noised_sin),
    })

    _df.plot()
    plt.show()
