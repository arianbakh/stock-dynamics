import arabic_reshaper
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
import warnings

from bidi.algorithm import get_display
from matplotlib import rc
from matplotlib.backends import backend_gtk3

from iran_stock import get_iran_stock_network
from settings import OUTPUT_DIR


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)


# File Settings
XI_PATH = os.path.join(OUTPUT_DIR, 'xi.npy')
LIBRARY_PATH = os.path.join(OUTPUT_DIR, 'library.npy')


# Algorithm Settings
RSI_PERIOD = 7
CV_DAYS = 14
TEST_DAYS = 21
SINDY_ITERATIONS = 10
CANDIDATE_LAMBDAS_RSI = [10 ** i for i in range(-9, -1)]  # empirical


def _exponential_moving_average(x, n):
    alpha = 1 / n
    s = np.zeros(x.shape)
    s[0] = x[0]  # this is automatically deep copy
    for i in range(1, s.shape[0]):
        s[i] = x[i] * alpha + s[i - 1] * (1 - alpha)
    return s


def _get_iran_stock_rsi():
    iran_stock_network = get_iran_stock_network()
    x = iran_stock_network.x
    u = x[1:] - x[:x.shape[0] - 1]
    u = u.clip(min=0)
    d = x[:x.shape[0] - 1] - x[1:]
    d = d.clip(min=0)
    rs = np.nan_to_num(_exponential_moving_average(u, RSI_PERIOD) / _exponential_moving_average(d, RSI_PERIOD))
    rsi = 100 - (100 / (1 + rs))
    return rsi, iran_stock_network.node_labels


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


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def _get_theta(x):  # empirical
    time_frames = x.shape[0] - 1

    column_list = [np.ones(time_frames)]
    library = [1]
    x_vectors = []
    for i in range(x.shape[1]):
        library.append((i,))
        x_vectors.append(x[:time_frames, i])
    column_list += x_vectors
    for subset in itertools.combinations(range(x.shape[1]), 2):
        library.append(subset)
        library.append(tuple(reversed(subset)))
        x_i = x[:time_frames, subset[0]]
        x_j = x[:time_frames, subset[1]]
        column_list.append(x_i / (1 + np.abs(x_j)))
        column_list.append(x_j / (1 + np.abs(x_i)))

    theta = np.column_stack(column_list)
    return library, theta


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


def _get_xi_and_library(normalized_rsi):
    if os.path.exists(XI_PATH) and os.path.exists(LIBRARY_PATH):
        library = np.load(LIBRARY_PATH, allow_pickle=True)
        xi_sindy = np.load(XI_PATH, allow_pickle=True)
    else:
        entire_x_dot = _get_x_dot(normalized_rsi)
        library, entire_theta = _get_theta(normalized_rsi)
        np.save(LIBRARY_PATH, library)
        test_index = entire_x_dot.shape[0] - TEST_DAYS
        x_dot_train = entire_x_dot[:test_index]
        theta_train = entire_theta[:test_index]
        xi_sindy = _optimum_sindy(x_dot_train, theta_train, CANDIDATE_LAMBDAS_RSI)
        np.save(XI_PATH, xi_sindy)
    return xi_sindy, library


def _predict_rsi(normalized_rsi, normalization_parameters, xi):
    test_index = normalized_rsi.shape[0] - TEST_DAYS
    normalized_rsi_hat = np.copy(normalized_rsi)
    for time_frame in range(test_index, normalized_rsi.shape[0]):
        library_hat, theta_hat = _get_theta(normalized_rsi_hat[time_frame - 1:time_frame + 1])
        x_dot_hat = np.matmul(theta_hat, xi.T)
        normalized_rsi_hat[time_frame] = normalized_rsi_hat[time_frame - 1] + x_dot_hat
    rsi_hat = _revert_x(normalized_rsi_hat, normalization_parameters)
    return rsi_hat


def _calculate_g(rsi_part):
    g = np.zeros((rsi_part.shape[1], rsi_part.shape[1]))
    for i in range(rsi_part.shape[1]):
        avg_xi_2 = np.mean(np.square(rsi_part[:, i]))
        for j in range(rsi_part.shape[1]):
            if i == j:
                g[i, j] = 1
            else:
                g[i, j] = np.mean(rsi_part[:, i] * rsi_part[:, j]) / avg_xi_2
    return g


def _calculate_impact(g):
    impact = np.zeros(g.shape[0])
    for i in range(g.shape[0]):
        impact[i] = np.mean(g.T[i, :])
    return impact


def _calculate_stability(g):
    stability = np.zeros(g.shape[0])
    for i in range(g.shape[0]):
        stability[i] = 1 / np.mean(g[i, :])
    return stability


def _draw_distribution(data, x_label, y_label, title, file_name):
    rc('font', weight=600)
    plt.subplots(figsize=(10, 10))
    ax = sns.distplot(
        data,
        bins=np.arange(np.min(data), np.max(data), (np.max(data) - np.min(data)) / 10),
        norm_hist=True
    )
    ax.set_title(get_display(arabic_reshaper.reshape(title)), fontsize=28, fontweight=500)
    ax.set_xlabel(get_display(arabic_reshaper.reshape(x_label)), fontsize=20, fontweight=500)
    ax.set_ylabel(get_display(arabic_reshaper.reshape(y_label)), fontsize=20, fontweight=500)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3, length=10, labelsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, file_name))
    plt.close('all')


def _better_label(complete_node_label):
    return get_display(arabic_reshaper.reshape(complete_node_label.replace('_', '-').split('-')[-1]))


def _save_table(data1, data2, labels, table_name):
    with open(os.path.join(OUTPUT_DIR, 'table_%s.txt' % table_name), 'w') as table_file:
        for i in range(len(data1)):
            table_file.write('%s & %.2f & %.2f \\\\\n' % (
                _better_label(labels[i]),
                data1[i],
                data2[i]
            ))


def run():
    rsi, node_labels = _get_iran_stock_rsi()
    normalized_rsi, normalization_parameters = _normalize_x(rsi)
    xi, library = _get_xi_and_library(normalized_rsi)
    predicted_rsi = _predict_rsi(normalized_rsi, normalization_parameters, xi)
    g1 = _calculate_g(predicted_rsi[-2 * TEST_DAYS:-TEST_DAYS])
    impact1 = _calculate_impact(g1)
    stability1 = _calculate_stability(g1)
    g2 = _calculate_g(predicted_rsi[-TEST_DAYS:])
    impact2 = _calculate_impact(g2)
    stability2 = _calculate_stability(g2)
    _draw_distribution(impact1, 'تأثیر', 'چگالی', 'مقدار تأثیر قبل از پیش‌بینی', 'impact_before_prediction.png')
    _draw_distribution(impact2, 'تأثیر', 'چگالی', 'مقدار تأثیر پس از پیش‌بینی', 'impact_after_prediction.png')
    _draw_distribution(stability1, 'ثبات', 'چگالی', 'مقدار ثبات قبل از پیش‌بینی', 'stability_before_prediction.png')
    _draw_distribution(stability2, 'ثبات', 'چگالی', 'مقدار ثبات پس از پیش‌بینی', 'stability_after_prediction.png')
    _save_table(impact1, impact2, node_labels, 'impact')
    _save_table(stability1, stability2, node_labels, 'stability')


if __name__ == '__main__':
    run()
