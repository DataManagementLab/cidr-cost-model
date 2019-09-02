#! /bin/bash

BIN=ops_test
COMPILER=g++
OPT="-std=c++17 ops_test.cpp -o $BIN"

echo $COMPILER $OPT
$COMPILER $OPT
success=$?

if [[ $success -eq 0 ]]
then
    echo "Created binary file $BIN"
    echo "Execute $(pwd)/$BIN to run the tests."
fi

exit $success

