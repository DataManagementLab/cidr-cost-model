#! /bin/bash

BIN=physical_cost_exp1
COMPILER=g++
OPT="-std=c++17 physical_cost_exp1.cpp -o $BIN -O3"

echo $COMPILER $OPT
$COMPILER $OPT
success=$?

if [[ $success -eq 0 ]]
then
    echo "Created binary file $BIN"
    echo "Execute $(pwd)/$BIN to run the tests."
fi

exit $success

