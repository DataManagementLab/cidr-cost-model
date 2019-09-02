import argparse
import csv
import os
import pickle

import math
import matplotlib.pyplot as plt
import pandas as pd
from util import figsize
from white_box_single_pipeline import build_probe_op
from white_box_single_pipeline import scan_branched_op, generate_test_and_train


def predict_simple_scan(train_size):
    model_name = f'models/white_box/training_data_share/scan_larger_160k_train_{train_size}.pickle'
    with open(model_name, 'rb') as handle:
        white_box_params = pickle.load(handle)

    return lambda df_row: scan_branched_op(params=white_box_params, selectivity=df_row['selectivity'],
                                           tuple_width=df_row['tuplewidth'], no_predicates=df_row['attrInPred'],
                                           no_tuples=df_row['tablerows'], tablesize=df_row['tablerows']).item()


def predict_query_runtime(train_size):
    with open(f'models/white_box/probe_train_{train_size}.pickle', 'rb') as handle:
        probe_params = pickle.load(handle)

    with open(f'models/white_box/build_train_{train_size}.pickle', 'rb') as handle:
        build_params = pickle.load(handle)

    return lambda df_row: 0.8 * (build_probe_op(params=probe_params, selectivity=df_row['selectivity'],
                                                tuple_width=df_row['tuplewidth'], no_predicates=df_row['attrInPred'],
                                                no_tuples=df_row['tablerows'], tablesize=df_row['tablerows']).item() +
                                 build_probe_op(params=build_params, selectivity=df_row['selectivity'],
                                                tuple_width=df_row['tuplewidth'], no_predicates=df_row['attrInPred'],
                                                no_tuples=df_row['tablerows'], tablesize=df_row['tablerows']).item())


def predict_probe_runtime(train_size):
    with open(f'models/white_box/probe_train_{train_size}.pickle', 'rb') as handle:
        probe_params = pickle.load(handle)

    return lambda df_row: 0.8 * (build_probe_op(params=probe_params, selectivity=df_row['selectivity'],
                                                tuple_width=df_row['tuplewidth'], no_predicates=df_row['attrInPred'],
                                                no_tuples=df_row['tablerows'], tablesize=df_row['tablerows']).item())


def predict_build_runtime(train_size):
    with open(f'models/white_box/build_train_{train_size}.pickle', 'rb') as handle:
        build_params = pickle.load(handle)

    return lambda df_row: 0.8 * (build_probe_op(params=build_params, selectivity=df_row['selectivity'],
                                                tuple_width=df_row['tuplewidth'], no_predicates=df_row['attrInPred'],
                                                no_tuples=df_row['tablerows'], tablesize=df_row['tablerows']).item())


def q_error(df_row):
    return max(df_row['white_box_estimate'] / df_row['runtime'], df_row['runtime'] / df_row['white_box_estimate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot_path_prefix', type=str, default='./plots/white_box/data_efficiency')
    parser.add_argument('--csv_path', type=str, default='data/simple_sel_pipeline.csv')
    parser.add_argument('--target_path', type=str, default='./plots/white_box/white_box_data_efficiency.csv')
    parser.add_argument('--train_sizes', help="Share of training data to train the black box model",
                        nargs='+', type=int, default=[1, 2, 5, 10, 20, 40, 60, 80])
    parser.add_argument('--min_table_size', type=int, default=160000)
    parser.add_argument('--max_table_size', type=int, default=math.inf)
    parser.add_argument('--train_test_split', type=float, default=0.9)
    parser.add_argument('--model', default='simple_scan')

    args = parser.parse_args()

    SAVE_PLOTS = True
    csv_rows = []

    if args.model == 'simple_scan':
        predict = predict_simple_scan
    elif args.model == 'join_query':
        predict = predict_query_runtime
    elif args.model == 'probe':
        predict = predict_probe_runtime
    elif args.model == 'build':
        predict = predict_build_runtime
    else:
        raise NotImplementedError("Unknown white box model.")

    # loop over training data sizes
    for train_size in args.train_sizes:
        print(f"Evaluating for train_size {train_size}")

        plot_path = f'{args.plot_path_prefix}/train_size_{train_size}'
        os.makedirs(plot_path, exist_ok=True)

        df = pd.read_csv(args.csv_path)
        df['white_box_estimate'] = df.apply(predict(train_size), axis=1)

        # do the plotting
        if SAVE_PLOTS:
            for ts in df.tablesizekb.unique():
                for tw in df.tuplewidth.unique():
                    if ts < args.min_table_size:
                        continue
                    df_temp_big = df.query("tuplewidth == " + str(tw) + " & tablesizekb == " + str(ts))
                    averaged_data = df_temp_big[['attrInPred', 'selectivity', 'runtime', 'white_box_estimate']].groupby(
                        ['attrInPred', 'selectivity']).mean().reset_index()
                    for attrInPred in averaged_data['attrInPred'].unique():
                        plot_data = averaged_data[averaged_data.attrInPred == attrInPred]

                        plt.figure(figsize=figsize())
                        plt.title(f"tablesizes={ts}, tuplewidth={tw}, no_attributes={attrInPred}")
                        plt.plot(plot_data['selectivity'], plot_data['runtime'], label="runtime")
                        plt.plot(plot_data['selectivity'], plot_data['white_box_estimate'], label="white box")
                        plt.legend()

                        plot_filename = "tablesize_{}_tuplewidth_{}_pred_{}.png"
                        plot_full_path = os.path.join(plot_path, plot_filename.format(ts, tw, attrInPred))
                        plt.savefig(plot_full_path, bbox_inches='tight')
                        plt.close()

        # generate error on test set
        _, test_data = generate_test_and_train(args.csv_path, args.min_table_size, args.max_table_size,
                                               args.train_test_split)
        df = pd.DataFrame(data=test_data,
                          columns=['tablerows', 'tablesizekb', 'selectivity', 'tuplewidth', 'attrInPred', 'runtime'])
        df['white_box_estimate'] = df.apply(predict(train_size), axis=1)
        df['q_error'] = df.apply(q_error, axis=1)

        csv_rows.append({'train_size': train_size,
                         'q_error_mean': df['q_error'].mean(),
                         'q_error_median': df['q_error'].median(),
                         'q_error_90_percentile': df['q_error'].quantile(0.9),
                         })

    print(f"Saving baseline results to {args.target_path}")
    with open(args.target_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)
