import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import warnings

from bidi.algorithm import get_display
from matplotlib.backends import backend_gtk3

from iran_stock import get_iran_stock_network
from settings import OUTPUT_DIR


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)
sns.set()


PERIOD = 14


def _exponential_moving_average(x, n):
    alpha = 1 / n
    s = np.zeros(x.shape)
    s[0] = x[0]  # this is automatically deep copy
    for i in range(1, s.shape[0]):
        s[i] = x[i] * alpha + s[i - 1] * (1 - alpha)
    return s


def _get_iran_stock_indicators():
    iran_stock_network = get_iran_stock_network()
    x = iran_stock_network.x
    u = x[1:] - x[:x.shape[0] - 1]
    u = u.clip(min=0)
    d = x[:x.shape[0] - 1] - x[1:]
    d = d.clip(min=0)
    rs = np.nan_to_num(_exponential_moving_average(u, PERIOD) / _exponential_moving_average(d, PERIOD))
    rsi = 100 - (100 / (1 + rs))
    stochastic_rsi = np.zeros(rsi.shape)
    for i in range(PERIOD - 1, stochastic_rsi.shape[0]):
        min_rsi = rsi[i - PERIOD + 1:i + 1].min(axis=0)
        max_rsi = rsi[i - PERIOD + 1:i + 1].max(axis=0)
        stochastic_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi)
    stochastic_rsi = np.nan_to_num(np.delete(stochastic_rsi, list(range(PERIOD - 1)), 0))
    return rsi, stochastic_rsi, iran_stock_network.node_labels


def _create_time_series(rsi, stochastic_rsi, node_labels):
    for node_id in range(rsi.shape[1]):
        ax = sns.lineplot(data=rsi[:, node_id])
        node_instrument_id, node_name = node_labels[node_id].split('_')
        reshaped_name = arabic_reshaper.reshape(node_name)
        name_display = get_display(reshaped_name)
        ax.set(xlabel='Days', ylabel='RSI', title=name_display)
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_%s_rsi.png' % (node_id, node_instrument_id)))
        plt.close('all')

    for node_id in range(stochastic_rsi.shape[1]):
        ax = sns.lineplot(data=stochastic_rsi[:, node_id])
        node_instrument_id, node_name = node_labels[node_id].split('_')
        reshaped_name = arabic_reshaper.reshape(node_name)
        name_display = get_display(reshaped_name)
        ax.set(xlabel='Days', ylabel='Stochastic RSI', title=name_display)
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_%s_stochastic_rsi.png' % (node_id, node_instrument_id)))
        plt.close('all')


def run():
    rsi, stochastic_rsi, node_labels = _get_iran_stock_indicators()
    _create_time_series(rsi, stochastic_rsi, node_labels)


if __name__ == '__main__':
    run()
