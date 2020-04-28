import math
import numpy as np
import sys

from iran_stock import get_iran_stock_network


# Algorithm Settings
SINDY_ITERATIONS = 10
MIN_PAST_DAYS = 8
MAX_PAST_DAYS = 10
MIN_FUTURE_DAYS = 4
MAX_FUTURE_DAYS = 6
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


def run():
    iran_stock_network = get_iran_stock_network()
    x = iran_stock_network.x

    x_dot = _get_x_dot(x)
    theta = _get_theta(x)

    for past in range(MIN_PAST_DAYS, MAX_PAST_DAYS + 1):
        for future in range(MIN_FUTURE_DAYS, MAX_FUTURE_DAYS + 1):
            print('%d past, %d future' % (past, future))
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

                future_x = np.copy(x[today:today + 1])
                future_x_dot = None
                for _ in range(future):
                    future_theta = _get_theta(future_x)
                    future_x_dot = np.matmul(future_theta, xi.T)
                    future_x += future_x_dot
                success += (np.sign(future_x_dot[0]) == np.sign(x_dot[today + future - 1]))
            success_rate = success / total_predictions
            print()  # newline
            print(success_rate)


if __name__ == '__main__':
    run()
