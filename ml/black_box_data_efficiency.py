import argparse
import csv
import glob
import json
import os
import pickle
from os import path
from parser import Parser

import matplotlib.pyplot as plt
import pandas as pd
from black_box import predict_cost
from util import figsize


def load_plotting_data(filename_prefix, no_operators, no_sample_tuples, dim_predicate_embedding):
    """
    Load data from json file and split it into training and test
    (generate two random distinct query lists with sizes based on given split point)

    :param filename_prefix: file prefix to load the data from
    :type filename_prefix: str
    :param train_split_fraction: which fraction of the queries to use for training
    :type train_split_fraction: float
    :param no_operators:
    :type no_operators:
    :param no_sample_tuples:
    :type no_sample_tuples:
    :return: two lists with training and test queries as well as meta information (numbers of columns and tables)
    :rtype: list[(OperatorNode, int)], list[(OperatorNode, int)], int, int
    """
    parser = Parser(no_operators, no_sample_tuples, dim_predicate_embedding)
    queries, num_colums, num_tables, _ = parser.parse_file(filename_prefix + ".json")

    # normalize by table sizes otherwise it does not work
    table_sizes = pd.read_csv(filename_prefix + ".csv")['tablerows']
    query_trees = [top_operator for (top_operator, runtime, index) in queries]

    return query_trees, table_sizes


def q_error(df_row):
    return max(df_row['black_box_estimate'] / df_row['runtime'], df_row['runtime'] / df_row['black_box_estimate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_operators', type=int, default=3)
    parser.add_argument('--no_sample_tuples', type=int, default=4)
    parser.add_argument('--dim_predicate_embedding', type=int, default=3,
                        help="Dimensions to represent the predicates")

    parser.add_argument('--base_path', type=str, default='./')
    parser.add_argument('--experiment_name', type=str, default='simple_sel_pipeline')
    parser.add_argument('--train_sizes', help="Share of training data to train the black box model",
                        nargs='+', type=int, default=[2, 4, 6, 8, 10, 20, 40, 80, 100])
    parser.add_argument('--min_table_size', type=int, default=160000)
    parser.add_argument('--save_plots', type=bool, default=False)

    args = parser.parse_args()

    plot_filename = "tablesize_{}_tuplewidth_{}_pred_{}.png"

    input_filename_prefix = path.join(args.base_path, "data", args.experiment_name)
    models_filename_prefix = path.join(args.base_path, "models", "black_box", args.experiment_name)
    plots_path = path.join(args.base_path, "plots", "black_box", args.experiment_name)

    # Load data
    query_trees, table_sizes = load_plotting_data(input_filename_prefix,
                                                  args.no_operators,
                                                  args.no_sample_tuples,
                                                  args.dim_predicate_embedding)

    df = pd.read_csv(input_filename_prefix + ".csv")

    csv_rows = []
    for size in args.train_sizes:
        print(f"Evaluating for train_size {size}")
        try:
            # Get dump from highest available epoch
            highest_epoch = max(int(fn.split(".")[-2])
                                for fn
                                in glob.glob(f"{models_filename_prefix}.bbmodel.{str(size)}.epoch.*.pickle")
                                )
            with open(models_filename_prefix + f".bbmodel.{str(size)}.epoch.{str(highest_epoch)}.pickle", 'rb') as handle:
                params = pickle.load(handle)

            # Determine plot path & create output directory for plots if needed
            plot_path_size_path = path.join(plots_path, f"train_size_{size}")
            os.makedirs(plot_path_size_path, exist_ok=True)

            def estimate_per_row(df_row):
                idx = df_row.name
                table_size = table_sizes[df_row.name]
                return table_size * predict_cost(query_trees[idx], params).item()

            df['black_box_estimate'] = df.apply(estimate_per_row, axis=1)

            # compute
            with open(input_filename_prefix + ".traintest.json") as json_file:
                train_test_split = json.load(json_file)

            df['q_error'] = df.apply(q_error, axis=1)
            test_q_errors = df['q_error'][train_test_split['test']]

            csv_rows.append({'train_size': size,
                             'q_error_mean': test_q_errors.mean(),
                             'q_error_median': test_q_errors.median(),
                             'q_error_90_percentile': test_q_errors.quantile(0.9),
                             })

            if args.save_plots:
                tuple_widths = df.tuplewidth.unique()
                tablesizes = df.tablesizekb.unique()

                for ts in tablesizes:
                    if ts < args.min_table_size:
                        continue
                    for tw in tuple_widths:
                        df_temp_big = df.query("tuplewidth == " + str(tw) + " & tablesizekb == " + str(ts))
                        if len(df_temp_big.index) > 0:
                            averaged_data = df_temp_big[
                                ['attrInPred', 'selectivity', 'runtime', 'black_box_estimate']].groupby(
                                ['attrInPred', 'selectivity']).mean().reset_index()
                            for attrInPred in averaged_data['attrInPred'].unique():
                                plot_data = averaged_data[averaged_data.attrInPred == attrInPred]

                                plt.figure(figsize=figsize())
                                plt.title(f"tablesizes={ts}, tuplewidth={tw}, no_attributes={attrInPred}")
                                plt.plot(plot_data['selectivity'], plot_data['runtime'], label="runtime")
                                plt.plot(plot_data['selectivity'], plot_data['black_box_estimate'], label="black box")
                                plt.legend()

                                plt.savefig(path.join(plot_path_size_path, plot_filename.format(ts, tw, attrInPred)), bbox_inches='tight')
                                plt.close()

        except FileNotFoundError:
            print(f'No model found for size {size}')

    target_path = path.join(plots_path, f"{args.experiment_name}_data_efficiency.csv")
    print(f"Saving baseline results to {target_path}")
    with open(target_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)
