import argparse
import json
import pickle
import random
from copy import deepcopy
from os import path
from parser import Parser
from time import perf_counter

import autograd.numpy as np
import autograd.numpy.random as npr
import pandas as pd
from autograd import grad
from autograd.misc.optimizers import adam
from operator_node import PredicateOperator
# Training data
from util import leaky_relu, concat_and_multiply, sigmoid

from joblib import Parallel, delayed
import multiprocessing


def vary_query(operator_node, num_columns, tables_to_colums):
    """
    Vary a (sub-)query represented by the given operator node

    :param operator_node: query representation
    :type operator_node: OperatorNode
    :param num_columns: number of columns
    :type num_columns: int
    :param tables_to_colums: lookup table from table ids to valid column ids
    :type tables_to_colums: Dict[int, list[int]]
    """
    # Adapt predicates of this Operator Node
    if operator_node.original_predicates_list is not None and len(operator_node.original_predicates_list) > 0:
        valid_ids = tables_to_colums[operator_node.table_id]
        new_predicates = []
        # For every predicate...
        for (p_column_id, p_operator, p_value) in operator_node.original_predicates_list:
            # ... choose a new column id randomly...
            p_column_id = random.choice(valid_ids)
            new_predicates.append((p_column_id, p_operator, p_value))
        # ... and use the new predicates afterwards
        operator_node.set_predicates(new_predicates, num_columns)

    # Vary sub queries
    for child in operator_node.children:
        vary_query(child, num_columns, tables_to_colums)


def load_data(base_path, experiment_name, train_split_fraction, no_operators, no_sample_tuples, expansion_factor,
              dim_predicate_embedding, size_threshold):
    """
    Load data from json file and split it into training and test
    (generate two random distinct query lists with sizes based on given split point)

    :param base_path: base path to use
    type base_path: str
    :param experiment_name: name of the experiment -- will be used to construct file names
    :type experiment_name: str
    :param train_split_fraction: which fraction of the queries to use for training
    :type train_split_fraction: float
    :param no_operators:
    :type no_operators:
    :param no_sample_tuples:
    :type no_sample_tuples:
    :param expansion_factor: How many variations of each query should be created?
    :type expansion_factor: int
    :param dim_predicate_embedding:
    :type dim_predicate_embedding:
    :param size_threshold: minimum table size for a query to be considered
    :type size_threshold: int
    :return: two lists with training and test queries as well as meta information (numbers of columns and tables)
    :rtype: list[(OperatorNode, int)], list[(OperatorNode, int)], int, int
    """
    filename_prefix = path.join(base_path, "data", experiment_name)
    parser = Parser(no_operators, no_sample_tuples, dim_predicate_embedding)
    table_rows = pd.read_csv(filename_prefix + ".csv")['tablerows']
    table_sizes_kb = pd.read_csv(filename_prefix + ".csv")['tablesizekb']
    queries, num_colums, num_tables, tables_to_columns = parser.parse_file(filename_prefix + ".json")

    # Shuffle for train-test-split -- queries based on the same plan can only end in train or test set
    random.shuffle(queries)

    # Postprocess Queries
    expanded_queries = []
    for query in queries:
        q_operator_node, q_runtime, q_index = query

        # Only consider tables with size over a certain threshold
        if table_sizes_kb[q_index] >= size_threshold:
            # Normalize runtime by table sizes otherwise it does not work
            query = (q_operator_node, q_runtime / table_rows[q_index], q_index)

            # Expand/vary queries
            for i in range(expansion_factor):
                query_copy = deepcopy(query)
                vary_query(query_copy[0], num_colums, tables_to_columns)
                expanded_queries.append(query_copy)

    split_point = int(len(expanded_queries) * train_split_fraction)
    return expanded_queries[:split_point], expanded_queries[split_point:], num_colums, num_tables


def init_params(input_size, state_size,
                h1_neurons_estimation_layer,
                no_operators, dim_operator_embedding,
                no_tables, dim_table_embedding,
                no_sample_tuples, dim_sample_embedding,
                predicate_input_length, dim_pred_embedding,
                join_predicate_input_length,
                param_scale=1):
    def rp(*shape):
        return npr.randn(*shape) * param_scale

    return {
        # lstm network
        'init_g': rp(1, state_size),
        'init_r': rp(1, state_size),
        'w_forget': rp(input_size + state_size + 1, state_size),
        'w_k_1': rp(input_size + state_size + 1, state_size),
        'w_r': rp(input_size + state_size + 1, state_size),
        'w_k_2': rp(input_size + state_size + 1, state_size),

        # estimation network
        'w_est_h1': rp(state_size + 1, h1_neurons_estimation_layer),
        'w_est_final': rp(h1_neurons_estimation_layer + 1, 1),

        # embedding network
        'w_emd_op': rp(no_operators + 1, dim_operator_embedding),
        'w_emd_t': rp(no_tables + 1, dim_table_embedding),
        'w_emd_s': rp(no_sample_tuples + 1, dim_sample_embedding),
        'w_emd_p': rp(predicate_input_length + 1, dim_pred_embedding),
        'w_emd_jp': rp(join_predicate_input_length + 1, dim_pred_embedding)}


def predict_cost(operator, params):
    # apply lstm architecture to compute operator tree embedding
    _, r_out = predict_representation(operator, params)

    # estimation network to get costs
    hidden_layer = leaky_relu(r_out, params['w_est_h1'])
    return leaky_relu(hidden_layer, params['w_est_final'])


def predict_embedding(node, params):
    embedding_op = leaky_relu(node.operator_feature, params['w_emd_op'])
    embedding_table = leaky_relu(node.table_feature, params['w_emd_t'])
    embedding_sample = leaky_relu(node.sample_bitmap, params['w_emd_s'])

    embedding_predicate = np.zeros((1, node.dim_predicate_embedding))
    for predicate_vector in node.predicates_list:
        current_embedding = leaky_relu(predicate_vector, params['w_emd_p'])
        embedding_predicate = np.maximum(embedding_predicate, current_embedding)

    for predicate_vector in node.join_predicates_list:
        current_embedding = leaky_relu(predicate_vector, params['w_emd_jp'])
        embedding_predicate = np.maximum(embedding_predicate, current_embedding)

    embedding = np.concatenate((embedding_op, embedding_table, embedding_sample, embedding_predicate), axis=1)

    return embedding


def predict_representation(operator, params):
    def tree_lstm(input, g_in_left, g_in_right, r_in_left, r_in_right):
        g_in = 1 / 2 * (g_in_left + g_in_right)
        r_in = 1 / 2 * (r_in_left + r_in_right)
        f_t = sigmoid(concat_and_multiply(params['w_forget'], r_in, input))
        k_t_1 = sigmoid(concat_and_multiply(params['w_k_1'], r_in, input))
        r_t = np.tanh(concat_and_multiply(params['w_r'], r_in, input))
        k_t_2 = sigmoid(concat_and_multiply(params['w_k_2'], r_in, input))

        g_out = f_t * g_in + k_t_1 * r_t
        r_out = k_t_2 * np.tanh(g_out)

        return g_out, r_out

    # predict feature embedding for current operator
    feature_vector = predict_embedding(operator, params)

    # leaf node
    if len(operator.children) == 0:
        return tree_lstm(feature_vector, params['init_g'], params['init_g'], params['init_r'],
                         params['init_r'])

    g_in_left, r_in_left = predict_representation(operator.children[0], params)
    g_in_right, r_in_right = predict_representation(operator.children[1], params)

    return tree_lstm(feature_vector, g_in_left, g_in_right, r_in_left, r_in_right)


def save_params(params, model_path):
    with open(model_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_train_test_split(base_path, experiment_name, train_queries, test_queries):
    """
    Save indices of train and test queries into a json file

    :param base_path: base path to use
    type base_path: str
    :param experiment_name: name of the experiment -- will be used to construct file names
    :type experiment_name: str
    :param train_queries: list of training queries
    :type train_queries: List[(OperatorNode, int, int, int)
    :param test_queries: list of test queries
    :type test_queries: List[(OperatorNode, int, int, int)
    """
    filename_prefix = path.join(base_path, "data", experiment_name)

    train_indices = list(set(query[2] for query in train_queries))
    test_indices = list(set(query[2] for query in test_queries))

    with open(filename_prefix + ".traintest.json", "w") as tt_json:
        json.dump({"train": train_indices, "test": test_indices}, tt_json, indent=2)


def accuracy(params, batch):
    error = 0
    for top_operator, runtime, _ in batch:
        cost_estimate = predict_cost(top_operator, params)
        error += np.square(cost_estimate - runtime)
    return error


def fit_model(size, training_data_full, args, input_dim_lstm, no_tables, predicate_input_length, join_predicate_input_length):
    full_size = len(training_data_full)
    size_split_point = int(full_size * size / 100)
    print(f"Training model with {size}% of the training data ({size_split_point}/{full_size} queries)")
    training_data = training_data_full[:size_split_point]

    initial_params = init_params(input_size=input_dim_lstm, state_size=args.dim_lstm_hidden,
                                 no_operators=args.no_operators, dim_operator_embedding=args.dim_operator_embedding,
                                 no_tables=no_tables, dim_table_embedding=args.dim_table_embedding,
                                 no_sample_tuples=args.no_sample_tuples,
                                 dim_sample_embedding=args.dim_sample_embedding,
                                 predicate_input_length=predicate_input_length,
                                 dim_pred_embedding=args.dim_predicate_embedding,
                                 join_predicate_input_length=join_predicate_input_length,
                                 h1_neurons_estimation_layer=args.dim_est_hidden,
                                 param_scale=args.param_scale)

    num_batches = int(np.ceil(len(training_data) / args.batch_size))

    model_filename_prefix = path.join(args.base_path, "models", "black_box", args.experiment_name)

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * args.batch_size, (idx + 1) * args.batch_size)


    def objective(params, iter):
        idx = batch_indices(iter)
        return accuracy(params, training_data[idx])

    start_t = perf_counter()

    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, training_data).item()
            test_acc = accuracy(params, test_data).item()
            training_time = perf_counter() - start_t

            epoch = iter // num_batches
            print(f"{size}: {epoch:15}{train_acc:20.10f}{test_acc:20.10f} {training_time:20.2f}s")
            if epoch > 0 and epoch % args.write_epochs == 0:
                save_params(params, model_filename_prefix + f".bbmodel.{str(size)}.epoch.{str(epoch)}.pickle")

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    objective_grad = grad(objective)

    optimized_params = adam(objective_grad, initial_params, num_iters=args.num_epochs * num_batches,
                            callback=print_perf, step_size=args.step_size)

    print(optimized_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_operators', type=int, default=3)
    parser.add_argument('--dim_operator_embedding', type=int, default=3,
                        help="Dimensions to represent the operator type of a node")
    parser.add_argument('--dim_table_embedding', type=int, default=3,
                        help="Dimensions to represent the table of a leave node")
    parser.add_argument('--no_sample_tuples', type=int, default=4)
    parser.add_argument('--dim_sample_embedding', type=int, default=3,
                        help="Dimensions to represent the sample tuples of a table")
    parser.add_argument('--dim_predicate_embedding', type=int, default=3,
                        help="Dimensions to represent the predicates")
    parser.add_argument('--dim_lstm_hidden', type=int, default=2, help="No of neurons in the lstm state (G_t)")
    parser.add_argument('--dim_est_hidden', type=int, default=3, help="No of neurons in the hidden layer of the "
                                                                      "estimation network")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--param_scale', type=float, default=0.1)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--base_path', type=str, default='./')
    parser.add_argument('--experiment_name', type=str, default='simple_sel_pipeline')
    parser.add_argument('--train_split_fraction', type=float, default=0.9)
    parser.add_argument('--table_size_threshold', type=int, default=160000)
    parser.add_argument('--train_sizes', help="Share of training data to train the black box model",
                        nargs='+', type=int, default=[2, 4, 6, 8, 10, 20, 40, 80, 100])
    parser.add_argument('--run_parallel', type=bool, default=False)

    parser.add_argument('--expansion_factor', type=int, default=10)
    parser.add_argument('--write_epochs', type=int, default=50)

    args = parser.parse_args()

    random.seed(42)

    # Load data
    training_data, test_data, no_columns, no_tables = load_data(args.base_path,
                                                                args.experiment_name,
                                                                args.train_split_fraction,
                                                                args.no_operators,
                                                                args.no_sample_tuples,
                                                                args.expansion_factor,
                                                                args.dim_predicate_embedding,
                                                                args.table_size_threshold)
    print(f"Training with {len(training_data)} training data points")
    print(f"Training with {len(test_data)} testing data points")

    save_train_test_split(args.base_path, args.experiment_name, training_data, test_data)

    # how many input neurons are required for tree lstm cell, i.e. length of embedding
    input_dim_lstm = args.dim_operator_embedding + args.dim_table_embedding + args.dim_sample_embedding + \
                     args.dim_predicate_embedding
    # predicate feature is column id plus operator id plus literal
    predicate_input_length = no_columns + len(PredicateOperator) + 1
    join_predicate_input_length = no_columns + len(PredicateOperator) + no_columns

    training_data_full = training_data

    print("Size      Epoch     |    Train accuracy  |       Test accuracy  |  Tranining Time")

    # Fit model
    if args.run_parallel:
        num_cores = min(multiprocessing.cpu_count(), len(args.train_sizes))
        print(f"Computing model using {num_cores} CPUs")
        Parallel(n_jobs=num_cores)(
            delayed(fit_model)(size, training_data_full, args, input_dim_lstm, no_tables, predicate_input_length, join_predicate_input_length) for size in args.train_sizes)
    else:
        for size in args.train_sizes:
            fit_model(size, training_data_full, args, input_dim_lstm, no_tables, predicate_input_length, join_predicate_input_length)
