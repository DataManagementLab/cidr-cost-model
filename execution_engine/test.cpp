#include "execution_engine.hpp"
#include "table.hpp"
#include "query.hpp"
#include "predicate.hpp"
#include "selection_ops.hpp"

struct SomeEngineConfig
{
    using Table_t = StandardTable;

    static const uint64_t numTables = 1;
};

template<typename Table_t>
struct QueryTest: public Query<Table_t>
{
    void execute(Table_t &table)
    {
        uint64_t column0 = 0;
        auto&& res0 = selection_op_branch(table.begin(column0), table.end(column0), PredicateTrue<typename Table_t::const_iterator>());
        //auto&& res2 = selection_op<PredicateAnd<1>>(table[column2]);

        std::cout << "Query result:\n";
        for(const auto &e : *res0)
        {
            std::cout << e << "\n";
        }
    }
};

int
main(int, char **)
{
    using Engine = ExecutionEngine<SomeEngineConfig>;
    using Table_t = typename Engine::Table_t;
    Engine engine({"../data/test.tbl"}, {});

    auto numColumns = engine.getTable(0).numColumns();
    size_t table_size = 100;
    std::vector<uint32_t> data;
    data.reserve(numColumns*table_size);

    for(uint64_t i = 0; i < numColumns*table_size; ++i)
    {
        data.emplace_back(i);
    }

    engine.getTable(0).loadFromMemory(data.data(), table_size);

    engine.execute<QueryTest<Table_t>>(0UL);
}
