import json

from operator_node import OperatorNode, PredicateOperator, OperatorType

# Not all types are implemented -- whenever a placeholder string is given to OperatorNode, this will cause an
# AttributeError -- such queries will be ignored
PRED_TYPES = ["Invalid", PredicateOperator.GREATER, PredicateOperator.LOWER, PredicateOperator.EQUALS, "True", "Half"]


class Parser:
    """
    Parser for json format

    General format:

    List of queries
    |-- Query: Meta Data + Pipeline
       |-- Pipeline: Runtime + List of operators
          |-- Operator: Scan, Build or Probe Node.

    Probe Nodes consume the previous entry of the operator list as left child
    and contain another pipeline as right child when translating to a tree.

    Example:
            JOIN
        SCAN    SCAN


                op2 (probe)
        op1 (scan)      op2-2 (build)
                        op2-1 (scan)


        pipeline: {
            "operators": [
                {op1},
                {op2,
                    pipeline->operators: [
                        {op2-1},
                        {op2-2}
                    ]
                }
            ],
            runtime_us
        }
    """

    def __init__(self, no_operators, no_sample_tuples, dim_predicate_embedding):
        """
        Constructor

        :param no_operators:
        :param no_sample_tuples:
        """
        self.json_object = {}
        self.operator_nodes = []
        self.no_columns = 0
        self.no_tables = 0
        self.no_operators = no_operators
        self.no_sample_tuples = no_sample_tuples
        self.dim_predicate_embedding = dim_predicate_embedding

    def parse_file(self, filename):
        """
        Parse given file

        :param filename: name of the file to parse
        :type filename: str
        :return: parsed data (training_data, test_data)
        :rtype: list, list
        """
        with open(filename, "r") as json_file:
            self.json_object = json.load(json_file)
        return self._parse()

    def parse_json_string(self, json_string):
        """
        Parse given json string

        :param json_string: json string to parse
        :type json_string: str
        :return: parsed data (training_data, test_data)
        :rtype: list, list
        """
        self.json_object = json.loads(json_string)
        return self._parse()

    def _parse(self):
        """
        Do the real parsing (on internal json object)

        :return: List of Tuples of OperatorNodes representing a pipeline and their corresponding overall runtime & Meta data (numbers of columns and tables, mapping from tables to column ids)
        :rtype: list[(OperatorNode, int)], int, int, Dict[int, list[int]]
        """
        self.operator_nodes = []

        self.no_columns = self.json_object["meta"]["numColumns"]
        self.no_tables = self.json_object["meta"]["numTables"]

        tables_to_columns = {e[0]: e[1] for e in self.json_object["meta"]["tableToColumnMapping"]}

        queries = self.json_object["queries"]

        parsed_queries = []
        for i, raw_query in enumerate(queries):
            try:
                parsed_queries.append(self._parse_query(raw_query, i))
            except AttributeError:
                print(f"Could not parse query {i}")

        return parsed_queries, self.no_columns, self.no_tables, tables_to_columns

    def _parse_query(self, raw_query, index):
        """
        Parse given raw query

        :param raw_query: json object encoding a query (pipeline)
        :type raw_query: Object
        :param index: index of this query in the list of queries (for later identification)
        :type index: int
        :return: Tuple of OperatorNode representing a pipeline and its corresponding overall runtime as well as its index in the list of queries
        :rtype: (OperatorNode, int, int)
        """
        runtime = raw_query["runtime_us"]

        raw_operators = list(raw_query["pipeline"]["operators"])
        raw_operators.reverse()
        op_node = self._parse_operator(raw_operators)

        return (op_node, runtime, index)

    def _parse_operator(self, raw_operators):
        """
        Parse list of operators from raw format (recursively if needed)

        :param raw_operators: list of raw operators
        :type raw_operators: list
        :return: OperatorNode representing the first relevant opertor in raw list
        :rtype: OperatorNode
        """
        raw_operator = raw_operators[0]
        if "sample_vector" in raw_operator and len(raw_operator["sample_vector"]) >= self.no_sample_tuples:
            sample_bitmap = raw_operator["sample_vector"][:self.no_sample_tuples]
        else:
            sample_bitmap = None

        # Probe (Join)
        if raw_operator["op_type"] == 2:
            # Generate join predicate (should be exactly one condition)
            assert len(raw_operator["join_predicates"]) == 1
            join_predicate_raw = raw_operator["join_predicates"][0]
            join_predicate = (join_predicate_raw["column_id"],
                              PRED_TYPES[join_predicate_raw["pred_type"]+1],
                              join_predicate_raw["column_id_build"])

            # = Get children recursively =
            # Left child is previous (next because of reverting) entry in this operator list
            child_left = self._parse_operator(raw_operators[1:])

            # Right child is top operation node of inner pipeline
            inner_raw_operators = list(join_predicate_raw["pipeline"]["operators"])
            inner_raw_operators.reverse()
            child_right = self._parse_operator(inner_raw_operators)

            # Generate Operator Node
            return OperatorNode(
                children=[child_left, child_right],
                op_id=OperatorType.HASH_JOIN,
                table_id=raw_operator["table_id"],
                predicates_list=None,
                join_predicates_list=[join_predicate],
                sample_bitmap=sample_bitmap,
                no_columns=self.no_columns,
                no_tables=self.no_tables,
                no_operators=self.no_operators,
                no_sample_tuples=self.no_sample_tuples,
                dim_predicate_embedding=self.dim_predicate_embedding
            )
        # Scan
        elif raw_operator["op_type"] == 0:
            # Generate Operator Node
            return OperatorNode(
                children=[],
                op_id=OperatorType.SCAN,
                table_id=raw_operator["table_id"],
                predicates_list=self._parse_predicates(raw_operator["predicates"]),
                join_predicates_list=None,
                sample_bitmap=sample_bitmap,
                no_columns=self.no_columns,
                no_tables=self.no_tables,
                no_operators=self.no_operators,
                no_sample_tuples=self.no_sample_tuples,
                dim_predicate_embedding=self.dim_predicate_embedding
            )
        # Build or other unsupported operation types
        else:
            # Ignore this operator and continue with next element of the pipeline
            return self._parse_operator(raw_operators[1:])

    @staticmethod
    def _parse_predicates(raw_predicates):
        """
        Parse raw list of predicates into tuple format using Enum

        :param raw_predicates: json object list of raw predicate_entries
        :type list
        :return: parsed predicates in tuple format
        :rtype: list[(int, PredicateOperator, Any)]
        """
        return [
            (
                rp["column_id"],
                PRED_TYPES[rp["pred_type"]+1],
                rp["literal_normalized"]
            )
            for rp in raw_predicates]
