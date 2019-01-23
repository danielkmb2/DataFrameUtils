from random import random, randrange
from matplotlib import pyplot
import pandas as pd
import numpy as np


def random_int_series(size: int, amplitude=10) -> pd.Series:
    return pd.Series([randrange(amplitude) for _ in range(size)])


def random_walk_int_series(size: int, step_size=1) -> pd.Series:
    random_walk = list()
    step = lambda: -step_size if random() < 0.5 else step_size
    random_walk.append(step())
    for i in range(1, size):
        movement = step()
        value = random_walk[i-1] + movement
        random_walk.append(value)

    return pd.Series(random_walk)


def sin_series(size: int, frequency=5, amplitude=1) -> pd.Series:
    f = frequency
    x = np.arange(size)
    y = np.sin(2 * np.pi * f * x / size)
    y = y * amplitude

    return pd.Series(y)


def apply_normal_noise(series: pd.Series, noise_size=0.1) -> pd.Series:

    noise_generator = lambda size: np.random.normal(0, 1, size)
    noised_signal = noise_generator(len(series)) * noise_size + series

    return pd.Series(noised_signal)


if __name__ == '__main__':
    s = random_int_series(1000, amplitude=20)
    pyplot.plot(s)
    pyplot.show()

    s = random_walk_int_series(1000)
    pyplot.plot(s)
    pyplot.show()

    s = sin_series(1000)
    pyplot.plot(s)
    pyplot.show()

    s = sin_series(1000)
    s = apply_normal_noise(s)
    pyplot.plot(s)
    pyplot.show()
