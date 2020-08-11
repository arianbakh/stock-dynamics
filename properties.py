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
from scipy.integrate import RK45

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


def _draw_distance_time_series(ts, distances):
    data_frame = pd.DataFrame({
        'index': ts,
        'distances': distances,
    })
    rc('font', weight=600)
    plt.subplots(figsize=(20, 10))
    ax = sns.scatterplot(x='index', y='distances', data=data_frame)
    ax.set_title(get_display(arabic_reshaper.reshape('دینامیک آشوبناک')), fontsize=28, fontweight=500)
    ax.set_xlabel(get_display(arabic_reshaper.reshape('زمان')), fontsize=20, fontweight=500)
    ax.set_ylabel(get_display(arabic_reshaper.reshape('فاصله تا نقطه قبلی')), fontsize=20, fontweight=500)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3, length=10, labelsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, 'chaos.png'))
    plt.close('all')


def _show_chaos(xi, library, normalized_rsi):
    def f(t, y):
        library_values = []
        for i in range(len(library)):
            if type(library[i]) == int:
                library_values.append(library[i])
            elif len(library[i]) == 1:
                library_values.append(y[library[i]])
            else:
                library_values.append(y[library[i][0]] / (1 + abs(y[library[i][1]])))
        y_dot = np.matmul(np.array(library_values), xi.T)
        return y_dot

    t0 = 0
    max_t = 80
    ts = []
    y0 = normalized_rsi[0]
    ys = [y0]
    distances = []
    integrator = RK45(f, t0, y0, max_t)
    while integrator.status != 'finished':
        integrator.step()
        distance = np.linalg.norm(integrator.y - ys[-1])
        print(distance)
        ts.append(integrator.t)
        ys.append(integrator.y)
        distances.append(distance)

    _draw_distance_time_series(ts, distances)


def _better_label(complete_node_label):
    return get_display(arabic_reshaper.reshape(complete_node_label.replace('_', '-').split('-')[-1]))


def _draw_contributions(top_contributions, node_index, node_labels):
    terms = []
    contributions = []
    for item in top_contributions:
        contributions.append(item[0])
        if type(item[1]) == int:
            terms.append(item[1])
        elif len(item[1]) == 1:
            terms.append(_better_label(node_labels[item[1][0]]))
        else:
            terms.append('%s \n _______________ \n (1 + %s) ' % (
                _better_label(node_labels[item[1][0]]),
                _better_label(node_labels[item[1][1]]))
            )

    data_frame = pd.DataFrame({
        'terms': terms,
        'contributions': contributions,
    })
    rc('font', weight=600)
    plt.subplots(figsize=(20, 12))
    ax = sns.barplot(x='terms', y='contributions', data=data_frame)
    title = 'جملات پرتأثیر در دینامیک %s' % node_labels[node_index].replace('_', '-').split('-')[-1]
    ax.set_title(get_display(arabic_reshaper.reshape(title)), fontsize=28, fontweight=500)
    ax.set_xlabel(get_display(arabic_reshaper.reshape('جملات')), fontsize=20, fontweight=500)
    ax.set_ylabel(get_display(arabic_reshaper.reshape('مجموع تاثیر')), fontsize=20, fontweight=500)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3, length=10, labelsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_contributions_to_node_%d.png' % node_index))
    plt.close('all')


def _max_contribution_to_node(xi, library, normalized_rsi, node_index, node_labels):
    csv_file_path = os.path.join(OUTPUT_DIR, 'contributions_to_%d.csv' % node_index)

    def f(t, y):
        library_values = []
        for i in range(len(library)):
            if type(library[i]) == int:
                library_values.append(library[i])
            elif len(library[i]) == 1:
                library_values.append(y[library[i]])
            else:
                library_values.append(y[library[i][0]] / (1 + abs(y[library[i][1]])))
        library_values_array = np.array(library_values)
        y_dot = np.matmul(library_values_array, xi.T)
        # the file operation needs to be done because we can't return extra arguments in this function
        with open(csv_file_path, 'w') as csv_file:
            text = ','.join([str(item) for item in (library_values_array * xi[node_index]).tolist()])
            csv_file.write(text + '\n')
        return y_dot

    t0 = 0
    max_t = 20  # from the chaos plot, before the chaos starts
    y0 = normalized_rsi[0]
    integrator = RK45(f, t0, y0, max_t)
    while integrator.status != 'finished':
        integrator.step()
        print(integrator.t)

    contributions = np.zeros(len(library))
    with open(csv_file_path, 'r') as csv_file:
        for line in csv_file.readlines():
            if line.strip():
                contributions += np.array([abs(float(item)) for item in line.strip().split(',')])
    contribution_functions = []
    for i in range(len(library)):
        contribution_functions.append((contributions[i], library[i]))
    sorted_contributions = sorted(contribution_functions, key=lambda x: -x[0])

    _draw_contributions(sorted_contributions[:5], node_index, node_labels)


def run():
    rsi, node_labels = _get_iran_stock_rsi()
    normalized_rsi, normalization_parameters = _normalize_x(rsi)
    xi, library = _get_xi_and_library(normalized_rsi)

    _show_chaos(xi, library, normalized_rsi)
    for node_index in [0, 1, 3, 4, 10, 36, 37]:  # ones marked as good in thesis
        _max_contribution_to_node(xi, library, normalized_rsi, node_index, node_labels)


if __name__ == '__main__':
    run()
