import arabic_reshaper
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import warnings

from bidi.algorithm import get_display
from matplotlib.backends import backend_gtk3

from iran_stock import get_iran_stock_network
from settings import OUTPUT_DIR


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)
sns.set()


# Algorithm Settings
RSI_PERIOD = 14
CV_PERCENT = 0.2
TEST_PERCENT = 0.2
MIN_FOURIER_HARMONICS = 10
MAX_FOURIER_HARMONICS = 20
SINDY_ITERATIONS = 5
CANDIDATE_LAMBDAS = [10 ** -i for i in range(2, 10)]  # empirical


def _get_theta(x):
    time_frames = x.shape[0] - 1
    column_list = [np.ones(time_frames)]
    vectors = []
    for power in range(4):
        vectors += [np.sin(x[:time_frames, i] * 2 ** power) for i in range(x.shape[1])]
    # for subset in itertools.combinations(vectors, 1):
    #     column_list.append(subset[0])
    for subset in itertools.combinations(vectors, 2):
        column_list.append(subset[0] * subset[1])
    theta = np.column_stack(column_list)
    return theta


def _sindy(x_dot, theta, candidate_lambda):
    xi = np.zeros((x_dot.shape[1], theta.shape[1]))
    for i in range(x_dot.shape[1]):
        ith_derivative = x_dot[:, i]
        ith_xi = np.linalg.lstsq(theta, ith_derivative, rcond=None)[0]
        for j in range(SINDY_ITERATIONS):
            small_indices = np.flatnonzero(np.absolute(ith_xi) < candidate_lambda)
            big_indices = np.flatnonzero(np.absolute(ith_xi) >= candidate_lambda)
            ith_xi[small_indices] = 0
            ith_xi[big_indices] = np.linalg.lstsq(theta[:, big_indices], ith_derivative, rcond=None)[0]
        xi[i] = ith_xi
    return xi


def _optimum_sindy(x_dot, theta):
    cv_index = int((1 - CV_PERCENT) * x_dot.shape[0])
    x_dot_train = x_dot[:cv_index]
    x_dot_cv = x_dot[cv_index:]
    theta_train = theta[:cv_index]
    theta_cv = theta[cv_index:]

    least_cost = sys.maxsize
    best_xi = None
    for i, candidate_lambda in enumerate(CANDIDATE_LAMBDAS):
        # progress bar
        sys.stdout.write('\r[%d/%d]' % (i + 1, len(CANDIDATE_LAMBDAS)))
        sys.stdout.flush()

        xi = _sindy(x_dot_train, theta_train, candidate_lambda)
        complexity = np.count_nonzero(xi)
        x_dot_hat = np.matmul(theta_cv, xi.T)
        mse = np.square(x_dot_cv - x_dot_hat).mean()
        if complexity:  # zero would mean no statements
            cost = mse * complexity
            if cost < least_cost:
                least_cost = cost
                best_xi = xi
    print()  # newline
    return best_xi


def _least_squares(x_dot, theta):
    return np.linalg.lstsq(theta, x_dot, rcond=None)[0].T


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


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
    stochastic_rsi = np.zeros(rsi.shape)
    for i in range(RSI_PERIOD - 1, stochastic_rsi.shape[0]):
        min_rsi = rsi[i - RSI_PERIOD + 1:i + 1].min(axis=0)
        max_rsi = rsi[i - RSI_PERIOD + 1:i + 1].max(axis=0)
        stochastic_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi)
    stochastic_rsi = np.nan_to_num(np.delete(stochastic_rsi, list(range(RSI_PERIOD - 1)), 0))
    return rsi, stochastic_rsi, iran_stock_network.node_labels


def _create_time_series(rsi, rsi_hat_fourier, rsi_hat_lstsq, rsi_hat_sindy, stochastic_rsi, node_labels):
    for node_id in range(rsi.shape[1]):
        data_frame = pd.DataFrame({
            'index': np.arange(rsi.shape[0]),
            'rsi': rsi[:, node_id],
            'rsi_hat_fourier': rsi_hat_fourier[:, node_id],
            'rsi_hat_lstsq': rsi_hat_lstsq[:, node_id],
            'rsi_hat_sindy': rsi_hat_sindy[:, node_id],
        })
        melted_data_frame = pd.melt(
            data_frame,
            id_vars=['index'],
            value_vars=['rsi', 'rsi_hat_fourier', 'rsi_hat_lstsq', 'rsi_hat_sindy']
        )
        ax = sns.lineplot(x='index', y='value', hue='variable', style='variable', data=melted_data_frame)
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


def _fourier_extrapolation(x, prediction_time_frames, fourier_harmonics):
    detrended_x, detrending_parameters = _get_detrended_x(x)
    columns = []
    for node_index in range(detrended_x.shape[1]):
        x_i = detrended_x[:, node_index]
        x_i_frequency_domain = np.fft.fft(x_i)
        time_frames = x_i.size
        frequencies = np.fft.fftfreq(time_frames)
        indexes = list(range(time_frames))
        indexes.sort(key=lambda index: np.absolute(frequencies[index]))
        t = np.arange(0, time_frames + prediction_time_frames)
        extrapolated_x_i = np.zeros(t.size)
        for i in indexes[:1 + fourier_harmonics * 2]:
            amplitude = np.absolute(x_i_frequency_domain[i]) / time_frames
            phase = np.angle(x_i_frequency_domain[i])
            extrapolated_x_i += amplitude * np.cos(2 * np.pi * frequencies[i] * t + phase)
        columns.append(extrapolated_x_i + detrending_parameters[node_index] * t)
    extrapolated_x = np.column_stack(columns)
    return extrapolated_x


def _optimized_fourier_extrapolation(x, prediction_time_frames):
    cv_index = int((1 - CV_PERCENT) * x.shape[0])
    x_test = x[:cv_index]
    x_cv = x[cv_index:]
    least_mse_cv = sys.maxsize
    best_fourier_harmonics = None
    for fourier_harmonics in range(MIN_FOURIER_HARMONICS, MAX_FOURIER_HARMONICS + 1):
        extrapolated_x_test = _fourier_extrapolation(x_test, x.shape[0] - cv_index, fourier_harmonics)
        mse_cv = np.square(x_cv - extrapolated_x_test[cv_index:]).mean()
        if mse_cv < least_mse_cv:
            least_mse_cv = mse_cv
            best_fourier_harmonics = fourier_harmonics
    return _fourier_extrapolation(x, prediction_time_frames, best_fourier_harmonics)


def run():
    rsi, stochastic_rsi, node_labels = _get_iran_stock_indicators()

    normalized_rsi, normalization_parameters = _normalize_x(rsi)

    entire_x_dot_rsi = _get_x_dot(normalized_rsi)
    entire_theta_rsi = _get_theta(normalized_rsi)
    test_index_rsi = int((1 - TEST_PERCENT) * entire_x_dot_rsi.shape[0])
    x_dot_rsi_train = entire_x_dot_rsi[:test_index_rsi]
    theta_rsi_train = entire_theta_rsi[:test_index_rsi]

    print('Calculating fourier predictions...')
    rsi_hat_fourier = _revert_x(
        _optimized_fourier_extrapolation(normalized_rsi[:test_index_rsi], normalized_rsi.shape[0] - test_index_rsi),
        normalization_parameters
    )
    rsi_hat_fourier[:test_index_rsi] = np.nan  # to avoid drawing

    print('Training lstsq...')
    xi_lstsq_rsi = _least_squares(x_dot_rsi_train, theta_rsi_train)
    # TODO find suitable library
    x_dot_test_rsi = entire_x_dot_rsi[test_index_rsi:]
    theta_test_rsi = entire_theta_rsi[test_index_rsi:]
    mse_test = np.square(np.matmul(theta_test_rsi, xi_lstsq_rsi.T) - x_dot_test_rsi).mean()
    print(mse_test)

    print('Training SINDy...')
    xi_sindy_rsi = _optimum_sindy(x_dot_rsi_train, theta_rsi_train)

    print('Creating lstsq and SINDy predictions...')
    normalized_rsi_hat_lstsq = np.copy(normalized_rsi)
    normalized_rsi_hat_sindy = np.copy(normalized_rsi)
    for time_frame in range(test_index_rsi, rsi.shape[0]):
        theta_hat_lstsq_rsi = _get_theta(normalized_rsi_hat_lstsq[time_frame - 1:time_frame + 1])
        x_dot_hat_lstsq_rsi = np.matmul(theta_hat_lstsq_rsi, xi_lstsq_rsi.T)
        normalized_rsi_hat_lstsq[time_frame] = normalized_rsi_hat_lstsq[time_frame - 1] + x_dot_hat_lstsq_rsi

        theta_hat_sindy_rsi = _get_theta(normalized_rsi_hat_sindy[time_frame - 1:time_frame + 1])
        x_dot_hat_sindy_rsi = np.matmul(theta_hat_sindy_rsi, xi_sindy_rsi.T)
        normalized_rsi_hat_sindy[time_frame] = normalized_rsi_hat_sindy[time_frame - 1] + x_dot_hat_sindy_rsi
    rsi_hat_lstsq = _revert_x(normalized_rsi_hat_lstsq, normalization_parameters)
    rsi_hat_lstsq[:test_index_rsi] = np.nan  # to avoid drawing

    rsi_hat_sindy = _revert_x(normalized_rsi_hat_sindy, normalization_parameters)
    rsi_hat_sindy[:test_index_rsi] = np.nan  # to avoid drawing

    print('Creating time series...')
    _create_time_series(rsi, rsi_hat_fourier, rsi_hat_lstsq, rsi_hat_sindy, stochastic_rsi, node_labels)


if __name__ == '__main__':
    run()
