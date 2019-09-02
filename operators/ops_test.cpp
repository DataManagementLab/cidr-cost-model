#include "selection_ops.hpp"
#include "../execution_engine/predicate.hpp"
#include "../storage/table.hpp"
#include <stdint.h>
#include <list>
#include <vector>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include <fstream>
#include <sstream>
#include <sys/stat.h>

static
Schema getSchema( )
{
    Schema schema = Column(0);
    return schema;
}

int main(int, char const**)
{
    const static unsigned int table_size = 100;
    uint32_t data[table_size];

    for (size_t i = 0; i < table_size; i++)
    {
        data[i] = i;
    }
    std::random_shuffle(data, data + table_size);

    const StandardTable table(getSchema( ), data, table_size);

    typedef PredicateTrue<StandardTable::iterator> Predicate_t;
    
    Predicate_t pred;
    // auto a = stor.begin();
    auto list = selection_op_branch(table.begin(0), table.end(0), pred);

    for(size_t i = 0; i < list->size(); ++i)
    {
        std::cout << list->operator[](i);
    }    
    std::cout << std::endl;

    return 0;
}
