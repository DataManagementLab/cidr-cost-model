#pragma once

#include <vector>

#include "../storage/table.hpp"
#include "../execution_engine/predicate.hpp"
#include "dump_types.hpp"
#include "../utils/cidr_exception.hpp"

#include <nlohmann/json.hpp>
using nlohmann::json;

struct query_stats
{
    uint64_t type;
    uint64_t total_runtime_us;
    json pipeline;
    //! table size in kb
    uint64_t tablesize;
    double selectivity;
    //! number of attributes
    uint64_t tuplewidth;
    //! attributes evaluated in the selection operator
    uint64_t attrInPred;
};

struct meta_stats
{
    uint64_t numTables;
    uint64_t numColumns;
    std::vector<std::pair<uint32_t,std::vector<uint32_t>>> tableToColumnMapping;
};

void to_json(json &j, const meta_stats &m)
{
    j = {{"numTables", m.numTables}, {"numColumns", m.numColumns}, {"tableToColumnMapping", m.tableToColumnMapping}};
}

void to_json(json &j, const query_stats &q) noexcept
{
    if(q.selectivity < -0.1 && q.selectivity > 1.1)
        THROW_CIDR_EXCEPTION("Expected -0.1 <= selectivity <= 1.1, but got selectivity=" << q.selectivity);

    j = json{{"plan_type", q.type},{"runtime_us", q.total_runtime_us}, {"pipeline", q.pipeline}, {"tablesize", q.tablesize}, {"selectivity", q.selectivity}, {"tuplewidth", q.tuplewidth}, {"attributesInPredicate", q.attrInPred}};
}

void to_json(json &j, const std::pair<uint64_t,json> &p)
{
    if(!p.second.empty())
        j = json{{"column_id", p.first},{"build_pipeline", p.second}};
}
