import arabic_reshaper
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import warnings

from bidi.algorithm import get_display
from matplotlib import rc
from matplotlib.backends import backend_gtk3

from iran_stock import get_iran_stock_network
from settings import OUTPUT_DIR, XI_PATH


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# Algorithm Settings
RSI_PERIOD = 7
CV_DAYS = 14
TEST_DAYS = 21
FOURIER_HARMONICS = 10
SINDY_ITERATIONS = 10
CANDIDATE_LAMBDAS_RSI = [10 ** i for i in range(-9, -1)]  # empirical
CANDIDATE_LAMBDAS_SRSI = list(np.arange(0.001, 0.011, 0.001))  # empirical


def _simple_moving_average(x, n):
    s = np.zeros(x.shape)
    s[:n - 1] = x[:n - 1]  # this is automatically deep copy
    for i in range(n - 1, s.shape[0]):
        s[i] = np.mean(x[i - (n - 1):i + 1], axis=0)
    return s


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
    rs = np.nan_to_num(_exponential_moving_average(u, RSI_PERIOD) / _exponential_moving_average(d, RSI_PERIOD))
    rsi = 100 - (100 / (1 + rs))
    srsi = np.zeros(rsi.shape)
    for i in range(RSI_PERIOD - 1, srsi.shape[0]):
        min_rsi = rsi[i - RSI_PERIOD + 1:i + 1].min(axis=0)
        max_rsi = rsi[i - RSI_PERIOD + 1:i + 1].max(axis=0)
        srsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi)
    srsi = np.nan_to_num(np.delete(srsi, list(range(RSI_PERIOD - 1)), 0))
    return rsi, srsi, iran_stock_network.node_labels


def _normalize_x(x):
    normalized_columns = []
    normalization_parameters = []
    for column_index in range(x.shape[1]):
        column = x[:, column_index]
        std = max(10 ** -9, np.std(column))  # to avoid division by zero
        mean = np.mean(column)
        normalized_column = (column - mean) / std
        normalized_columns.append(normalized_column)
        normalization_parameters.append((mean, std))
    normalized_x = np.column_stack(normalized_columns)
    return normalized_x, normalization_parameters


def _revert_x(normalized_x, normalization_parameters):
    reverted_columns = []
    for column_index in range(normalized_x.shape[1]):
        column = normalized_x[:, column_index]
        mean, std = normalization_parameters[column_index]
        reverted_column = column * std + mean
        reverted_columns.append(reverted_column)
    reverted_x = np.column_stack(reverted_columns)
    return reverted_x


def _get_detrended_x(x):
    columns = []
    detrending_parameters = []
    for node_index in range(x.shape[1]):
        x_i = x[:, node_index]
        t = np.arange(0, x_i.shape[0])
        linear_trend = np.polyfit(t, x_i, 1)[0]
        detrending_parameters.append(linear_trend)
        column = x_i - linear_trend * t
        columns.append(column)
    detrended_x = np.column_stack(columns)
    return detrended_x, detrending_parameters


def _fourier_extrapolation(x, prediction_time_frames):
    detrended_x, detrending_parameters = _get_detrended_x(x)
    columns = []
    for node_index in range(detrended_x.shape[1]):
        x_i = detrended_x[:, node_index]
        x_i_frequency_domain = np.fft.fft(x_i)
        time_frames = x_i.size
        frequencies = np.fft.fftfreq(time_frames)
        amplitudes = np.absolute(x_i_frequency_domain) / time_frames
        phases = np.angle(x_i_frequency_domain)
        amplitude_threshold = np.copy(amplitudes)
        amplitude_threshold.sort()
        amplitude_threshold = amplitude_threshold[-FOURIER_HARMONICS]
        t = np.arange(0, time_frames + prediction_time_frames)
        extrapolated_x_i = np.zeros(t.size)
        for i in range(time_frames):
            amplitude = amplitudes[i]
            if amplitude >= amplitude_threshold:
                frequency = frequencies[i]
                phase = phases[i]
                extrapolated_x_i += amplitude * np.cos(2 * np.pi * frequency * t + phase)
        columns.append(extrapolated_x_i + detrending_parameters[node_index] * t)
    extrapolated_x = np.column_stack(columns)
    return extrapolated_x


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def _get_theta(x):  # empirical
    time_frames = x.shape[0] - 1

    x_vectors = [x[:time_frames, i] for i in range(x.shape[1])]
    column_list = [np.ones(time_frames)] + x_vectors
    for subset in itertools.combinations(x_vectors, 2):
        column_list.append(subset[0] / (1 + np.abs(subset[1])))
        column_list.append(subset[1] / (1 + np.abs(subset[0])))

    theta = np.column_stack(column_list)
    return theta


def _least_squares(x_dot, theta):
    return np.linalg.lstsq(theta, x_dot, rcond=None)[0].T


def _single_node_sindy(x_dot_i, theta, candidate_lambda):
    xi_i = np.linalg.lstsq(theta, x_dot_i, rcond=None)[0]
    for j in range(SINDY_ITERATIONS):
        small_indices = np.flatnonzero(np.absolute(xi_i) < candidate_lambda)
        big_indices = np.flatnonzero(np.absolute(xi_i) >= candidate_lambda)
        xi_i[small_indices] = 0
        xi_i[big_indices] = np.linalg.lstsq(theta[:, big_indices], x_dot_i, rcond=None)[0]
    return xi_i


def _optimum_sindy(x_dot, theta, candidate_lambdas):
    cv_index = x_dot.shape[0] - CV_DAYS
    x_dot_train = x_dot[:cv_index]
    x_dot_cv = x_dot[cv_index:]
    theta_train = theta[:cv_index]
    theta_cv = theta[cv_index:]

    xi = np.zeros((x_dot_train.shape[1], theta_train.shape[1]))
    for i in range(x_dot_train.shape[1]):
        # progress bar
        sys.stdout.write('\rNode [%d/%d]' % (i + 1, x_dot_train.shape[1]))
        sys.stdout.flush()

        least_cost = sys.maxsize
        best_xi_i = None
        x_dot_i = x_dot_train[:, i]
        x_dot_cv_i = x_dot_cv[:, i]
        for candidate_lambda in candidate_lambdas:
            xi_i = _single_node_sindy(x_dot_i, theta_train, candidate_lambda)
            complexity = math.log(1 + np.count_nonzero(xi_i))
            x_dot_hat_i = np.matmul(theta_cv, xi_i.T)
            mse_cv = np.square(x_dot_cv_i - x_dot_hat_i).mean()
            if complexity:  # zero would mean no statements
                cost = mse_cv * complexity
                if cost < least_cost:
                    least_cost = cost
                    best_xi_i = xi_i
        xi[i] = best_xi_i
    print()  # newline
    return xi


def thresholding_alg(y, lag, threshold, influence):
    """
    slightly modified the code from:
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/43512887#43512887
    """
    signals = np.zeros(len(y))
    filtered_y = np.array(y)
    avg_filter = np.zeros(len(y))
    std_filter = np.zeros(len(y))
    avg_filter[lag - 1] = np.mean(y[0:lag])
    std_filter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avg_filter[i - 1]) > threshold * std_filter[i - 1]:
            if y[i] > avg_filter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filtered_y[i] = influence * y[i] + (1 - influence) * filtered_y[i - 1]
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1):i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1):i + 1])
        else:
            signals[i] = 0
            filtered_y[i] = y[i]
            avg_filter[i] = np.mean(filtered_y[(i - lag + 1):i + 1])
            std_filter[i] = np.std(filtered_y[(i - lag + 1):i + 1])

    return signals


def _draw_time_series(
        indicator,
        indicator_name,
        indicator_hat_fourier,
        indicator_hat_lstsq,
        indicator_hat_sindy,
        node_labels,
        test_index):
    for node_id in range(indicator.shape[1]):
        # Time Series plot
        data_frame = pd.DataFrame({
            'index': np.arange(indicator.shape[0]),
            indicator_name: indicator[:, node_id],
            'Fourier': indicator_hat_fourier[:, node_id],
            # 'Least_Squares': indicator_hat_lstsq[:, node_id],
            'SINDy': indicator_hat_sindy[:, node_id],
        })
        melted_data_frame = pd.melt(
            data_frame,
            id_vars=['index'],
            value_vars=[
                indicator_name,
                'Fourier',
                # 'Least-Squares',
                'SINDy',
            ]
        )
        rc('font', weight=600)
        plt.subplots(figsize=(20, 10))
        ax = sns.lineplot(x='index', y='value', hue='variable', style='variable', data=melted_data_frame, linewidth=4)
        node_instrument_id, node_name = node_labels[node_id].split('_')
        ax.set_title(get_display(arabic_reshaper.reshape(node_name)), fontsize=28, fontweight=500)
        ax.set_xlabel(get_display(arabic_reshaper.reshape('روزها')), fontsize=20, fontweight=500)
        ax.set_ylabel(indicator_name, fontsize=20, fontweight=500)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3, length=10, labelsize=16)
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_%s_%s.png' % (node_id, node_instrument_id, indicator_name)))
        plt.close('all')

        # Signal Plot
        lag = 5
        threshold = 1
        influence = 1
        indicator_signal = thresholding_alg(indicator[:, node_id], lag, threshold, influence)[test_index:]
        indicator_hat_sindy_signal = \
            thresholding_alg(indicator_hat_sindy[:, node_id], lag, threshold, influence)[test_index:]
        data_frame = pd.DataFrame({
            'index': np.arange(len(indicator_signal)),
            '%s' % indicator_name + get_display(arabic_reshaper.reshape('قله‌های')): indicator_signal,
            'SINDy' + get_display(arabic_reshaper.reshape('قله‌های')): indicator_hat_sindy_signal,
        })
        melted_data_frame = pd.melt(
            data_frame,
            id_vars=['index'],
            value_vars=[
                '%s' % indicator_name + get_display(arabic_reshaper.reshape('قله‌های')),
                'SINDy' + get_display(arabic_reshaper.reshape('قله‌های')),
            ]
        )
        rc('font', weight=600)
        plt.subplots(figsize=(20, 10))
        ax = sns.lineplot(x='index', y='value', hue='variable', style='variable', data=melted_data_frame, linewidth=4)
        node_instrument_id, node_name = node_labels[node_id].split('_')
        ax.set_title(get_display(arabic_reshaper.reshape(node_name)), fontsize=28, fontweight=500)
        ax.set_xlabel(get_display(arabic_reshaper.reshape('روزهای پیش‌بینی شده')), fontsize=20, fontweight=500)
        ax.set_ylabel(get_display(arabic_reshaper.reshape('قله‌های تشخیص داده شده')), fontsize=20, fontweight=500)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        ax.tick_params(width=3, length=10, labelsize=16)
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_%s_%s_peaks.png' % (node_id, node_instrument_id, indicator_name)))
        plt.close('all')


def _mean_absolute_error(test_data, prediction_data):
    return np.abs(test_data - prediction_data).mean()


def _create_indicator_time_series(indicator, indicator_name, node_labels, candidate_lambdas_indicator):
    normalized_indicator, normalization_parameters = _normalize_x(indicator)

    entire_x_dot = _get_x_dot(normalized_indicator)
    entire_theta = _get_theta(normalized_indicator)
    test_index = entire_x_dot.shape[0] - TEST_DAYS
    x_dot_train = entire_x_dot[:test_index]
    theta_train = entire_theta[:test_index]

    print('Calculating fourier predictions...')
    indicator_hat_fourier = _revert_x(
        _fourier_extrapolation(normalized_indicator[:test_index], normalized_indicator.shape[0] - test_index),
        normalization_parameters
    )
    indicator_hat_fourier[:test_index] = np.nan  # to avoid drawing

    print('Creating lstsq predictions...')
    xi_lstsq = _least_squares(x_dot_train, theta_train)

    normalized_indicator_hat_lstsq = np.copy(normalized_indicator)
    for time_frame in range(test_index, indicator.shape[0]):
        theta_hat_lstsq = _get_theta(normalized_indicator_hat_lstsq[time_frame - 1:time_frame + 1])
        x_dot_hat_lstsq = np.matmul(theta_hat_lstsq, xi_lstsq.T)
        normalized_indicator_hat_lstsq[time_frame] = normalized_indicator_hat_lstsq[time_frame - 1] + x_dot_hat_lstsq
    indicator_hat_lstsq = _revert_x(normalized_indicator_hat_lstsq, normalization_parameters)
    indicator_hat_lstsq[:test_index] = np.nan  # to avoid drawing

    print('MAE', _mean_absolute_error(
        normalized_indicator_hat_lstsq[test_index:],
        normalized_indicator[test_index:]
    ))

    print('Creating SINDy predictions...')
    if os.path.exists(XI_PATH):
        xi_sindy = np.load(XI_PATH, allow_pickle=True)
    else:
        xi_sindy = _optimum_sindy(x_dot_train, theta_train, candidate_lambdas_indicator)
        np.save(XI_PATH, xi_sindy)

    normalized_indicator_hat_sindy = np.copy(normalized_indicator)
    for time_frame in range(test_index, indicator.shape[0]):
        theta_hat_sindy = _get_theta(normalized_indicator_hat_sindy[time_frame - 1:time_frame + 1])
        x_dot_hat_sindy = np.matmul(theta_hat_sindy, xi_sindy.T)
        normalized_indicator_hat_sindy[time_frame] = normalized_indicator_hat_sindy[time_frame - 1] + x_dot_hat_sindy
    indicator_hat_sindy = _revert_x(normalized_indicator_hat_sindy, normalization_parameters)
    indicator_hat_sindy[:test_index] = np.nan  # to avoid drawing

    print('Drawing time series...')
    _draw_time_series(
        indicator,
        indicator_name,
        indicator_hat_fourier,
        indicator_hat_lstsq,
        indicator_hat_sindy,
        node_labels,
        test_index
    )


def run():
    rsi, srsi, node_labels = _get_iran_stock_indicators()
    _create_indicator_time_series(rsi, 'RSI', node_labels, CANDIDATE_LAMBDAS_RSI)
    # _create_indicator_time_series(srsi, 'SRSI', node_labels, CANDIDATE_LAMBDAS_SRSI)


if __name__ == '__main__':
    run()
