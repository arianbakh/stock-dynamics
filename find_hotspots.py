import arabic_reshaper
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
import warnings

from bidi.algorithm import get_display
from matplotlib.backends import backend_gtk3
from matplotlib.patches import Rectangle

from iran_stock import get_iran_stock_network
from settings import OUTPUT_DIR


warnings.filterwarnings('ignore', module=backend_gtk3.__name__)
sns.set()
IRAN_STOCK_NETWORK = get_iran_stock_network()


# Algorithm Settings
SINDY_ITERATIONS = 10
MAX_PAST_DAYS = 25
MAX_FUTURE_DAYS = 10
MAX_POWER = 2
LAMBDA_RANGE = [10 ** -14, 2 * 10 ** -2]  # empirical
LAMBDA_STEP = 2


# Calculated Settings
CANDIDATE_LAMBDAS = [
    LAMBDA_STEP ** i for i in range(
        int(math.log(abs(LAMBDA_RANGE[0])) / math.log(LAMBDA_STEP)),
        int(math.log(abs(LAMBDA_RANGE[1])) / math.log(LAMBDA_STEP))
    )
]


def _get_theta(x):
    time_frames = x.shape[0]
    column_list = [np.ones(time_frames)]
    for i in range(x.shape[1]):
        x_i = x[:time_frames, i]
        for power in range(1, MAX_POWER + 1):
            column_list.append(x_i ** power)
        for j in range(x.shape[1]):
            if j > i:
                x_j = x[:time_frames, j]
                for first_power in range(1, MAX_POWER + 1):
                    for second_power in range(1, MAX_POWER + 1):
                        column_list.append((x_i ** first_power) * (x_j ** second_power))
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
    least_cost = sys.maxsize
    best_xi = None
    for candidate_lambda in CANDIDATE_LAMBDAS:
        xi = _sindy(x_dot, theta, candidate_lambda)
        complexity = np.count_nonzero(xi)
        x_dot_hat = np.matmul(theta, xi.T)
        mse = np.square(x_dot - x_dot_hat).mean()
        if complexity:  # zero would mean no statements
            cost = mse * complexity
            if cost < least_cost:
                least_cost = cost
                best_xi = xi
    return best_xi


def _least_squares(x_dot, theta):
    return np.linalg.lstsq(theta, x_dot, rcond=None)[0].T


def _get_x_dot(x):
    x_dot = (x[1:] - x[:len(x) - 1])
    return x_dot


def _get_success_rates():
    x = IRAN_STOCK_NETWORK.x

    x_dot = _get_x_dot(x)
    theta = _get_theta(x)

    success_rates = np.zeros((MAX_PAST_DAYS + 1, MAX_FUTURE_DAYS + 1, x.shape[1]))  # includes 0
    for past in range(1, MAX_PAST_DAYS + 1):
        for future in range(1, MAX_FUTURE_DAYS + 1):
            # progress bar
            sys.stdout.write('\r[%d/%d]' % ((past - 1) * MAX_FUTURE_DAYS + future, MAX_PAST_DAYS * MAX_FUTURE_DAYS))
            sys.stdout.flush()

            cached_path = os.path.join(OUTPUT_DIR, '%d_%d_success_rate.p' % (past, future))
            if os.path.exists(cached_path):
                with open(cached_path, 'rb') as cached_file:
                    success_rate = pickle.load(cached_file)
            else:
                success = np.zeros(x.shape[1])
                total_predictions = x.shape[0] - future - past
                for today in range(past, x.shape[0] - future):
                    # progress bar
                    sys.stdout.write('\r[%d/%d]' % (today - past + 1, total_predictions))
                    sys.stdout.flush()

                    past_x_dot = x_dot[today - past:today]
                    past_theta = theta[today - past:today]
                    # xi = _optimum_sindy(past_x_dot, past_theta)  # TODO uncomment
                    xi = _least_squares(past_x_dot, past_theta)  # TODO remove

                    future_x = np.copy(x[today:today + 1])  # deep copy, because we will modify it
                    future_x_dot = None
                    for _ in range(future):
                        future_theta = _get_theta(future_x)
                        future_x_dot = np.matmul(future_theta, xi.T)
                        future_x += future_x_dot
                    success += (np.sign(future_x_dot[0]) == np.sign(x_dot[today + future - 1]))
                print()  # newline
                success_rate = success / total_predictions
                with open(cached_path, 'wb') as cached_file:
                    pickle.dump(success_rate, cached_file)
            success_rates[past, future] = success_rate
    print()  # newline
    return success_rates


def _create_heat_maps():
    success_rates = _get_success_rates() * 100  # convert to percent
    for node_id in range(success_rates.shape[2]):
        success_matrix = success_rates[1:, 1:, node_id]
        hot_spot = None
        max_success = 0
        for i in range(1, success_matrix.shape[0] - 1):
            for j in range(1, success_matrix.shape[1] - 1):
                if np.all(success_matrix[i-1:i+2, j-1:j+2] > 50) and success_matrix[i, j] > max_success:
                    hot_spot = (j, i)
                    max_success = success_matrix[i, j]
        data_set = pd.DataFrame(
            success_matrix,
            columns=list(range(1, success_rates.shape[1])),
            index=list(range(1, success_rates.shape[0]))
        )
        plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(data_set, annot=True, fmt=".1f", vmin=40, vmax=60, center=50, linewidths=3)
        if hot_spot:
            ax.add_patch(Rectangle(hot_spot, 1, 1, fill=False, edgecolor='blue', lw=5))
        node_instrument_id, node_name = IRAN_STOCK_NETWORK.node_labels[node_id].split('_')
        reshaped_name = arabic_reshaper.reshape(node_name)
        name_display = get_display(reshaped_name)
        ax.set(xlabel='Future Days', ylabel='Past Days', title=name_display)
        plt.savefig(os.path.join(OUTPUT_DIR, 'node_%d_%s_heat_map.png' % (node_id, node_instrument_id)))
        plt.close('all')


def run():
    _create_heat_maps()


if __name__ == '__main__':
    run()
