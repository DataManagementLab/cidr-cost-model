from enum import Enum

from autograd import numpy as np


class OperatorNode:

    def __init__(self, children, op_id, table_id=None, predicates_list=None, join_predicates_list=None,
                 sample_bitmap=None, no_columns=0, no_tables=0, no_operators=0, no_sample_tuples=0,
                 dim_predicate_embedding=1):
        self.children = children
        assert len(children) == 2 or len(children) == 0, "Operator tree must be a binary tree"

        self.operator_feature = np.zeros((1, no_operators))
        self.operator_feature[0, op_id.value] = 1
        self.dim_predicate_embedding = dim_predicate_embedding
        self.table_id = table_id

        self.table_feature = np.zeros((1, no_tables))
        if table_id is not None:
            self.table_feature[0, table_id] = 1

        if sample_bitmap is None:
            self.sample_bitmap = np.zeros((1, no_sample_tuples))
        else:
            self.sample_bitmap = np.array(sample_bitmap).reshape((1, no_sample_tuples))

        self.predicates_list = []
        self.original_predicates_list = predicates_list
        if predicates_list is not None:
            self.set_predicates(predicates_list, no_columns)

        # encode join predicates as vectors
        def encode_join_predicate(predicate):
            col_id, predicate_op, right_column_id = predicate
            column_vector = np.zeros((1, no_columns))
            column_vector[0, col_id] = 1

            op_vector = np.zeros((1, len(PredicateOperator)))
            op_vector[0, predicate_op.value] = 1

            right_column_vector = np.zeros((1, no_columns))
            right_column_vector[0, right_column_id] = 1

            return np.concatenate((column_vector, op_vector, right_column_vector), axis=1)

        self.join_predicates_list = []
        if join_predicates_list is not None:
            self.join_predicates_list = [encode_join_predicate(join_predicate) for join_predicate in
                                         join_predicates_list]

    def set_predicates(self, predicates_list, no_columns):
        self.predicates_list = [self.encode_predicate(predicate, no_columns) for predicate in predicates_list]

    # encode predicates as vectors
    def encode_predicate(self, predicate, no_columns):
        col_id, predicate_op, literal = predicate
        column_vector = np.zeros((1, no_columns))
        column_vector[0, col_id] = 1

        op_vector = np.zeros((1, len(PredicateOperator)))
        op_vector[0, predicate_op.value] = 1

        return np.concatenate((column_vector, op_vector, np.array([[literal]])), axis=1)


class PredicateOperator(Enum):
    EQUALS = 0
    LOWER = 1
    GREATER = 2


class OperatorType(Enum):
    SCAN = 0
    HASH_JOIN = 1
