import argparse
import logging
import os
import pickle
import time

import autograd.numpy.random as npr
import math
from autograd import numpy as np, grad
from autograd.misc.optimizers import adam


def generate_test_and_train(path, min_table_size, max_table_size, train_test_split, excluded_table_sizes=[]):
    runtime_data = np.genfromtxt(path, delimiter=',')
    # remove header and operator column
    runtime_data = runtime_data[1:, 0:6]

    # 0         1           2           3          4          5
    # tablerows,tablesizekb,selectivity,tuplewidth,attrInPred,runtime,operator
    runtime_data = runtime_data[(runtime_data[:, 1] >= min_table_size) & (runtime_data[:, 1] <= max_table_size)]

    for excluded_table_size in excluded_table_sizes:
        runtime_data = runtime_data[runtime_data[:, 1] != excluded_table_size]

    # usual train test split
    np.random.seed(42)
    np.random.shuffle(runtime_data)

    size_split_point = int(len(runtime_data) * train_test_split)
    test_data = runtime_data[size_split_point:]
    training_data = runtime_data[:size_split_point]

    return training_data, test_data


def init_params(pipeline, dim_weights, param_scale, dim_nn=3):
    def rp(*shape):
        return npr.randn(*shape) * param_scale

    if pipeline == 'scan':
        return {
            'w_slope_left': rp(dim_weights, 1),
            'w_intercept_left': rp(dim_weights, 1),
            'w_slope_right': rp(dim_weights, 1),
            'w_intercept_right': rp(dim_weights, 1),
            'w_final_weight_1': rp(1, 1),
            'w_final_weight_2': rp(1, 1)
        }
    elif pipeline == 'build' or pipeline == 'probe':
        return {
            'w_slope': rp(dim_weights, 1),
            'w_intercept': rp(dim_weights, 1),
            'w_final_weight': rp(1, 1),
            'w_final_constant': rp(1, 1),
            'w_final_slope': rp(1, 1),
            'w_final_intercept': rp(1, 1),
            'w_slope_h1': rp(dim_weights + 1, dim_nn),
            'w_slope_h2': rp(dim_nn + 1, dim_nn),
            'w_slope_final': rp(dim_nn + 1, 1),
            'w_intercept_h1': rp(dim_weights + 1, dim_nn),
            'w_intercept_h2': rp(dim_nn + 1, dim_nn),
            'w_intercept_final': rp(dim_nn + 1, 1)
        }
    else:
        raise NotImplementedError


def save_params(params, model_path):
    with open(model_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def scan_branched_op(params, selectivity, tuple_width, no_predicates, no_tuples, tablesize):
    """
    Model branched scan operator runtime as piecewise linear model. Slope and intercept of the linear model will
    depend on tuple width. This is not optimized but rather written for easy readability and changes.
    :return: runtime estimate
    """

    # slope and intercept depends linearly on tuple_width and no_predicates
    def linear_feature_combination(weight_key):
        return params[weight_key][0] * tuple_width + \
               params[weight_key][1] * no_predicates + \
               params[weight_key][2] * no_predicates / tuple_width + \
               params[weight_key][3]

    # piecewise linear model with maximum at selectivity of 0.5
    if selectivity < 0.5:
        slope = linear_feature_combination('w_slope_left')
        intercept = linear_feature_combination('w_intercept_left')
        cost_per_tuple = slope * selectivity + intercept
        return params['w_final_weight_1'] * no_tuples * cost_per_tuple
    else:
        slope = linear_feature_combination('w_slope_right')
        intercept = linear_feature_combination('w_intercept_right')
        cost_per_tuple = slope * selectivity + intercept
        return params['w_final_weight_2'] * no_tuples * cost_per_tuple


def build_probe_op(params, selectivity, tuple_width, no_predicates, no_tuples, tablesize):
    """
    Model build or probe hashtable operator runtime as linear model. This is not optimized but rather written for easy
    readability and changes.
    :return: runtime estimate
    """

    # relu like
    # def feature_combination(weight_key):
    #     x_0 = params[weight_key][3] * np.log(tablesize) + params[weight_key][4]
    #     x_1 = params[weight_key][5] * np.log(tablesize) * tuple_width + params[weight_key][6]
    #     return params[weight_key][0] * tuple_width + \
    #            np.where(x_0 > 0, x_0, x_0 * 0.001) + \
    #            np.where(x_1 > 0, x_1, x_1 * 0.001) + \
    #            params[weight_key][7]

    # linear
    def feature_combination(weight_key):
        return params[weight_key][0] * tuple_width + \
               params[weight_key][1] * np.log(tablesize) + \
               params[weight_key][2] * np.log(tablesize) * tuple_width + \
               params[weight_key][3] * no_predicates / tuple_width + \
               params[weight_key][4] * no_predicates + \
               params[weight_key][5]

    # neural network
    # def feature_combination(weight_key):
    #     # estimation network to get costs
    #     features = np.array([[np.log(tuple_width), np.log(tablesize)]])
    #     hidden_layer = leaky_relu(features, params[weight_key + '_h1'])
    #     hidden_layer = leaky_relu(hidden_layer, params[weight_key + '_h2'])
    #     return leaky_relu(hidden_layer, params[weight_key + '_final'])

    # piecewise linear model with maximum at selectivity of 0.5
    slope = params['w_final_slope'] * feature_combination('w_slope')
    intercept = params['w_final_intercept'] * feature_combination('w_intercept')
    cost_per_tuple = slope * selectivity + intercept
    return params['w_final_weight'] * no_tuples * cost_per_tuple - params['w_final_intercept']


def accuracy(params, batch, pipeline_cost_model):
    error = 0
    for x in batch:
        # 0         1           2           3          4          5
        # tablerows,tablesizekb,selectivity,tuplewidth,attrInPred,runtime,operator
        error += np.square(
            pipeline_cost_model(params=params, selectivity=x[2], tuple_width=x[3], no_predicates=x[4], no_tuples=x[0],
                                tablesize=x[1]) - x[5])
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--param_scale', type=float, default=0.1)
    parser.add_argument('--step_size', type=float, default=0.001)

    parser.add_argument('--training_data_path', default='data/white_box/simple_sel_pipeline.csv')
    parser.add_argument('--model_path_prefix', default='data/white_box/scan_larger_160k')
    parser.add_argument('--min_table_size', type=int, default=-math.inf)
    parser.add_argument('--max_table_size', type=int, default=math.inf)
    parser.add_argument('--excluded_table_sizes', help="Share of training data to train the black box model",
                        nargs='+', type=int, default=[])

    parser.add_argument('--train_test_split', type=float, default=0.9)
    parser.add_argument('--pipeline', default='scan')
    parser.add_argument('--train_sizes', help="Share of training data to train the black box model",
                        nargs='+', type=int, default=[100])

    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.pipeline, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    if args.pipeline == 'scan':
        pipeline_cost_model = scan_branched_op
    elif args.pipeline == 'probe' or args.pipeline == 'build':
        pipeline_cost_model = build_probe_op
    else:
        raise NotImplementedError('Unknown pipeline.')

    full_training_data, test_data = generate_test_and_train(args.training_data_path, args.min_table_size,
                                                            args.max_table_size, args.train_test_split,
                                                            args.excluded_table_sizes)

    for train_size in args.train_sizes:

        # tuple_width, no_predicates, no_tuples, tablesize
        # dim_weights = 4
        dim_weights = 6
        initial_params = init_params(pipeline=args.pipeline, dim_weights=dim_weights,
                                     param_scale=args.param_scale)

        # just a certain percentage of training_data
        size_split_point = int(len(full_training_data) * train_size / 100)
        training_data = full_training_data[:size_split_point]

        num_batches = int(np.ceil(len(training_data) / args.batch_size))
        num_epochs = int(args.num_epochs)
        logger.info(f"Training model with {train_size}% of the training data ({len(training_data)}) queries "
                    f"for {num_epochs} epochs")


        def batch_indices(iter):
            idx = iter % num_batches
            return slice(idx * args.batch_size, (idx + 1) * args.batch_size)


        def objective(params, iter):
            idx = batch_indices(iter)
            return accuracy(params=params, batch=training_data[idx], pipeline_cost_model=pipeline_cost_model)


        def callback_per_epoch(params, iter, gradient):
            epoch = iter // num_batches
            if iter % num_batches == 0 and epoch % 1 == 0:
                train_acc = accuracy(params=params, batch=training_data, pipeline_cost_model=pipeline_cost_model).item()
                test_acc = accuracy(params=params, batch=test_data, pipeline_cost_model=pipeline_cost_model).item()

                logger.info(f"Finished {epoch}/{num_epochs} epochs ({epoch / num_epochs * 100:.2f}%) "
                            f"for {train_size}% of the training data.")
                logger.info(f"\t\tTrain: {train_acc:20.10f}")
                logger.info(f"\t\tTest:  {test_acc:20.10f}")

                save_params(params, f'{args.model_path_prefix}_train_{train_size}.pickle')


        objective_grad = grad(objective)
        optimized_params = adam(objective_grad, initial_params, num_iters=num_epochs * num_batches,
                                callback=callback_per_epoch, step_size=args.step_size)
