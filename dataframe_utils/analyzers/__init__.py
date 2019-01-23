from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from luminol.anomaly_detector import AnomalyDetector
from luminol.correlator import Correlator

import sys
sys.path.append(".")
from dataframe_utils.analyzers.FeatureSelector import FeatureSelector


def plot_feature_importances(df: pd.DataFrame) -> None:
    fs = FeatureSelector(data=df)
    fs.identify_collinear(0)
    fs.plot_collinear()
    plt.show()


def correlate_series(s1: pd.Series, s2: pd.Series, window_len=41):
    ts1 = s1.to_dict()
    ts2 = s2.to_dict()

    results = np.zeros(len(s1))

    for i in range(len(s1) - window_len + 1):
        time_period = (i, i + window_len)
        correlator = Correlator(ts1, ts2, time_period=time_period)
        results[i] = correlator.correlation_result.coefficient

    return pd.Series(results)


def correlate_dataframe(df: pd.DataFrame, roi_col: str, window_len=41):
    correlations_df = pd.DataFrame()
    for col in [c for c in df.columns if c != roi_col]:
        correlations_df[col] = correlate_series(df[roi_col], df[col], window_len)

    return correlations_df


def print_correlation_resume(df: pd.DataFrame, roi_col: str, window_len=41):
    print("Correlation resume:")

    corr_df = correlate_dataframe(df, roi_col, window_len)
    for c in corr_df.columns:
        print()
        print("* " + c)
        print(" - Average correlation with " + roi_col + ": " + str(corr_df[c].mean()))
        print(" - Median correlation with " + roi_col + ": " + str(corr_df[c].median()))
        print(" - Std correlation with " + roi_col + ": " + str(corr_df[c].std()))

    print()
    print("Highest mean correlation: " + corr_df.mean().idxmax())
    print("Highest median correlation: " + corr_df.median().idxmax())
    print("Lowest std correlation: " + corr_df.std().idxmax())
    print()


def get_anomaly_series(series: pd.Series, algorithm: str = "bitmap_detector") -> pd.Series:
    assert algorithm in ["bitmap_detector", "derivative_detector", "exp_avg_detector"]

    ts = series.to_dict()
    detector = AnomalyDetector(ts, algorithm_name=algorithm)
    scores = detector.get_all_scores()
    scores = [s for _, s in scores.iteritems()]
    return pd.Series(scores)


def get_anomalies(series: pd.Series, algorithm: str = "bitmap_detector") -> List[dict]:
    assert algorithm in ["bitmap_detector", "derivative_detector", "exp_avg_detector"]

    ts = series.to_dict()
    detector = AnomalyDetector(ts, algorithm_name=algorithm)
    anomalies = detector.get_anomalies()
    return [{
        "start_time": _.start_timestamp,
        "end_time": _.end_timestamp,
        "top_score_time": _.exact_timestamp,
        "score": _.anomaly_score,
    } for _ in anomalies]


def plot_anomalies(series: pd.Series) -> None:
    _anomalies = get_anomalies(series, algorithm="bitmap_detector")
    plt.figure()
    plt.plot(series)

    for a in _anomalies:
        score = a["score"]
        start = a["start_time"]
        end = a["end_time"]
        plt.axvspan(start - 5, end - 1, facecolor='0.1', alpha=0.05 * score)
        plt.axvspan(start - 1, end + 1, facecolor='0.1', alpha=0.1 * score)
        plt.axvspan(start + 1, end + 5, facecolor='0.1', alpha=0.05 * score)
    plt.show()


def plot_df_correlations(df: pd.DataFrame) -> None:
    correlation_df = correlate_dataframe(df, roi_col="sin", window_len=41)
    correlation_df.plot()
    plt.show()

    plot_feature_importances(df)


def plot_anomaly_series(series: pd.Series) -> None:
    _anomalies = get_anomaly_series(series, algorithm="bitmap_detector")

    plt.figure()
    plt.plot(series)

    for i in range(0, len(_anomalies)):
        plt.axvspan(i, i + 5, facecolor='0.1', alpha=0.1 * _anomalies[i])
    plt.show()


def plot_series_decomposition(series: pd.Series, freq=1, model='additive') -> None:
    result = seasonal_decompose(series.values, freq=freq, model=model)
    _ = result.plot()
    plt.show()


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
    get_noised_sin = lambda size: pd.Series(sin + gen_noise(size))

    _df = pd.DataFrame({
        "sin": sin,
        "noised_sin1": get_noised_sin(500),
        "noised_sin2": get_noised_sin(500),
        "noised_sin3": get_noised_sin(500),
    })

    _df.plot()
    plt.show()

    # series correlation
    print_correlation_resume(_df, roi_col="sin", window_len=41)
    plot_df_correlations(_df)

    # anomaly by time series
    plot_anomaly_series(_df["noised_sin1"])

    # anomalous moments
    plot_anomalies(_df["noised_sin1"])

    # decomposition
    plot_series_decomposition(_df["noised_sin1"], freq=20)
