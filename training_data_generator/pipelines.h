#pragma once
#include <unordered_map>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

#include "../storage/table.hpp"
#include "../execution_engine/predicate.hpp"
#include "../execution_engine/dump_types.hpp"
#include "../utils/vector_helper.hpp"

#include <nlohmann/json.hpp>
using nlohmann::json;

// --------------------------- GLOBAL STUFF ---------------------------
typedef uint32_t data_type;

typedef Predicate<Table<data_type>::iterator> PredAbstract;
typedef PredicateTrue<Table<data_type>::iterator> PredTrue;
typedef PredicateHalf<Table<data_type>::iterator> PredHalf;
typedef PredicateGreaterThan<Table<data_type>::iterator> PredGreater;
typedef PredicateLessThan<Table<data_type>::iterator> PredLess;
typedef PredicateEqual<Table<data_type>::iterator> PredEqual;
typedef PredicateSelectivity<Table<data_type>::iterator> PredSel;
const static size_t num_tables = 27;
Table<data_type> tables[num_tables];

const static size_t mem_pool_size = 2'000'000'000; // 8gb
static size_t mem_pool_offset = 0;
data_type *memory_pool = (data_type*) aligned_alloc(64, mem_pool_size*sizeof(data_type));

const static size_t sample_vector_size = 100;

uint32_t to_abs_column_id(uint32_t table_id, uint32_t column_id)
{
    for(int64_t i = 0; i < table_id; ++i)
    {
        column_id += tables[i].numColumns();
    }
    return column_id;
}

std::vector<std::pair<uint32_t,std::vector<uint32_t>>> get_table_to_col_map(uint32_t table_cnt)
{
    std::vector<std::pair<uint32_t,std::vector<uint32_t>>> ids;
    uint32_t start_column = 0;
    for(int64_t i = 0; i < table_cnt; ++i)
    {
        std::vector<uint32_t> col_for_table;
        for (size_t j = 0; j < tables[i].numColumns(); j++)
        {
            col_for_table.push_back(j+start_column);
        }
        ids.push_back({i, col_for_table});

        start_column += tables[i].numColumns();
    }
    
    return ids;
}



Table<data_type>& abs_column_to_table(uint32_t abs_column_id, uint32_t &column_id)
{
    uint64_t pos = 0;

    for(uint64_t i = 0; i < num_tables; ++i)
    {
        pos += tables[i].numColumns();

        if (abs_column_id < pos)
        {
            column_id = abs_column_id - (pos - tables[i].numColumns());
            assert(column_id < tables[i].numColumns());
            return tables[i];
        }
    }

    throw std::runtime_error("Invalid abs column id!");

}

template<typename T>
double normalize_column_value(uint32_t abs_column, T value) noexcept
{
    uint32_t column_id = 0;
    auto& table = abs_column_to_table(abs_column, column_id);
    auto& stats = table.getStatistics(column_id);
    
    //! TODO \todo{Gather high and low of column}
    T high = stats.max;
    T low = stats.min;
    double ret;

    T range = std::abs(high - low);
    T offset;

    if(range)
    {
        
        if(std::abs(low) < std::abs(high))
        {
            offset = low;
        }
        else
        {
            offset = high;
        }
        
        if(offset < 0)
        {
            offset -= 1;
        }
        else
        {
            if(offset > 0)
            {
                offset += 1;
            }
        }
        
        range+=2;

        ret = (value - offset) / (range);
    }
    else
    {
        ret = 0.0;
    }
    

    if(ret < -0.1 && ret > 1.1)
        THROW_CIDR_EXCEPTION("Normalized value " << ret << " out range [-0.1,1.1]");

    return ret;
}

// -------------- INTERMEDIATE TUPLE REPRESENTATION --------------
struct Intermediate_tuple_t 
{
    Intermediate_tuple_t(){}
    Intermediate_tuple_t(std::initializer_list<std::pair<uint32_t, data_type*>> tuples)
    {
        uint32_t cols = 0;
        for (auto tup : tuples)
        {
            cols += tup.first;
        }

        data = memory_pool + mem_pool_offset;
        mem_pool_offset += cols;

        uint32_t col_offset = 0;
        for (auto tup : tuples)
        {
            std::copy(tup.second + col_offset, tup.second + col_offset + tup.first, data);
            col_offset += tup.first;
        }
    }

    Intermediate_tuple_t(const Intermediate_tuple_t &other)
    {
        if (other.data != nullptr)
        {
            data = memory_pool + mem_pool_offset;
            mem_pool_offset += other.num_cols;
            std::copy(other.data, other.data + other.num_cols, data);
        }
    }

    Intermediate_tuple_t &operator=(const Intermediate_tuple_t &other)
    {
        if (other.data != nullptr)
        {
            data = memory_pool + mem_pool_offset;
            mem_pool_offset += other.num_cols;
            std::copy(other.data, other.data + other.num_cols, data);
        }
        return *this;
    }

    const Intermediate_tuple_t& operator=(Intermediate_tuple_t&& other)
    {
        if (other.data != nullptr)
        {
            data = memory_pool + mem_pool_offset;
            mem_pool_offset += other.num_cols;
            std::copy(other.data, other.data + other.num_cols, data);
        }
        return *this;
    }

    ~Intermediate_tuple_t()
    {
    }
    uint32_t num_cols = 0;
    data_type* data = nullptr;
};


// ------------------------ OPERATOR STATISTICS ------------------------
struct op_stats
{
    using predicate_list_t = std::vector<std::pair<uint64_t,PredAbstract>>;

    //! Pair of absolute column_id (unique in DB) and predicate
    /**
    *   Must hold absolute column_id as table_id is required for conversion
    *   and this is not contained in the predicate!
    */
private:
    predicate_list_t predicates;

public:
    op_type op = op_type::INVALID_OP;
    int table_id = -1;
    size_t input_cardinality = 0;
    size_t output_cardinality = 0;
    
    std::pair<uint64_t,json> build_side = {0, {}};

    //! Sample vector - is deterministic which tuples are picked from each table. Only used for selection operator
    std::vector<int> sample_vector;

    op_stats() = default;

    //! _predicates: pair.first is relative column id!!
    op_stats(op_type _op, int64_t _table_id, std::vector<std::pair<uint64_t,PredAbstract>> _predicates):
        op(_op), table_id(_table_id)
    {

        for(const auto& p : _predicates)
        {
            add_predicate(p.first, p.second);
        }
    }

    op_stats(op_type _op, int64_t _table_id):
        op_stats(_op, _table_id, {})
    {}

    //! Converts column_id to absolute column id and then add abs. column id and predicate to predicates of operator
    void add_predicate(uint64_t column_id, const PredAbstract &pred)
    {
        predicates.emplace_back(to_abs_column_id(table_id,column_id), pred);
    }

    const predicate_list_t& getPredicates() const
    {
        return predicates;
    }

};

void to_json(json &j, const op_stats &s)
{
    // Probe operator
    if(!s.build_side.second.empty())
    {
        assert(s.op == PROBE);
        assert(s.getPredicates().size() == 1);

        const std::pair<uint64_t,PredAbstract> &p = s.getPredicates().at(0);
        json join_predicate = json{{"column_id", p.first},{"pred_type", p.second.getPredicateType()}};
        join_predicate.emplace("column_id_build", s.build_side.first);
        join_predicate.emplace("pipeline", s.build_side.second);
        json join_predicates = std::vector<json>{join_predicate};

        j = json{{"op_type", s.op}, {"table_id", s.table_id}, {"input_selectivity", s.input_cardinality}, {"output_selectivity", s.output_cardinality}};
        j.emplace("join_predicates", join_predicates);
    }
    else
    {
        j = json{{"op_type", s.op}, {"table_id", s.table_id}, {"input_selectivity", s.input_cardinality}, {"output_selectivity", s.output_cardinality}, {"predicates", s.getPredicates()}, {"sample_vector", s.sample_vector}};
    }
}

template <size_t predicate_count>
std::vector<int> get_sample_vector(uint32_t table_id, PredAbstract **preds)
{
    srand(1); //Seed random generator to be determistic!
    std::vector<int> sample_vector;
    sample_vector.reserve(sample_vector_size);
    auto &table = tables[table_id];
    auto iterator = table.begin(0);
    for (size_t i = 0; i < sample_vector_size; i++)
    {
        size_t rowid = rand() % table.numRows();
        iterator.pos = rowid;
        bool boolean = true;
        for (size_t i = 0; i < predicate_count; i++)
        {
            boolean &= preds[i]->operator()(iterator);
        }
        sample_vector.push_back(boolean ? 1 : 0);
    }
    return sample_vector;   
}

// --------------------------- PIPELINES ---------------------------
struct pipeline
{
    size_t runtime_us = 0; //Pipeline runtime

    struct std::vector<op_stats> stats{};

    virtual uint64_t getType() const
    {
        return INVALID_PIPELINE;
    }


    virtual json generate_training_data(const std::vector<std::pair<uint64_t,json>> &build_stats = {})
    {
        uint64_t hit = 0;
        for(auto&& s : stats)
        {
            if(s.op == PROBE && hit < build_stats.size())
            {
                s.build_side = build_stats.at(hit++);
            }
        }
        json j = json{{"runtime_us", runtime_us},{"operators", stats}};
        
#ifdef DEBUG
        // std::cout << j << std::endl;
#endif

        return j;
    }
};


struct pipeline_sel : public pipeline
{
template <bool generate_stats, size_t predicate_count>
    void inline run(uint32_t baseTable_id, PredAbstract **preds, std::vector<Intermediate_tuple_t> &output)
    {
        if constexpr(generate_stats)
        {
            stats.emplace_back(SELECT_BRANCH, baseTable_id);
            for (size_t i = 0; i < predicate_count; ++i)
                stats[0].add_predicate(preds[i]->column, *preds[i]);
        }

        // A bit unsafe, but stats are accessed when generate_stats == false
        struct op_stats &sel_op_stats = stats[0];

        // std::cout << "Executing pipeline_sel" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto table = tables[baseTable_id];
        for (auto it=table.begin(0); it!=table.end(0); ++it)
        {
            bool boolean = true;
            for (size_t i = 0; i < predicate_count; i++)
            {
                boolean &= preds[i]->operator()(it);
            }
            
            if (boolean)
            {
                if constexpr(generate_stats)
                    ++sel_op_stats.output_cardinality;

                Intermediate_tuple_t a({{it.getNumColumns(), it._data}});
                output.push_back(a);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        if constexpr(!generate_stats)
            runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if constexpr(generate_stats)
        {
            //Add sample_vector
            sel_op_stats.sample_vector = get_sample_vector<predicate_count>(baseTable_id, preds);
            sel_op_stats.input_cardinality = table.numRows();
        }
    }

    json generate_training_data(const std::vector<std::pair<uint64_t,json>> &build_stats = {}) override
    {
        struct op_stats &sel_op_stats = stats[0];

        std::cout << "----- pipeline_sel -----" << std::endl;
        std::cout << "sel_op_stats \n\t input_cardinality: " << sel_op_stats.input_cardinality << "\n\t output_cardinality: " << sel_op_stats.output_cardinality << std::endl;
        std::cout << "Runtime: " << runtime_us << " us" << std::endl;

        return pipeline::generate_training_data(build_stats);
    }
};

//! TODO \todo{Tiemo: @Lasse Branching select?}
struct pipeline_sel_build : public pipeline
{
    virtual uint64_t getType() const override
    {
        return SEL_BUILD;
    }

    template <bool generate_stats, size_t predicate_count>
    void inline run(uint32_t baseTable_id, std::unordered_map<data_type, Intermediate_tuple_t> &hashtable, uint32_t build_key, PredAbstract **preds)
    {
        if constexpr(generate_stats)
        {
            stats.emplace_back(SELECT_BRANCH, baseTable_id);
            for (size_t i = 0; i < predicate_count; ++i)
                stats[0].add_predicate(preds[i]->column, *preds[i]);
            stats.emplace_back(BUILD, baseTable_id);
        }

        // A bit unsafe, but stats are accessed when generate_stats == false
        struct op_stats &sel_op_stats = stats[0];
        struct op_stats &build_op_stats = stats[1];

        // std::cout << "Executing pipeline_sel_build" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto table = tables[baseTable_id];
        for (auto it=table.begin(build_key); it!=table.end(build_key); ++it)
        {
            bool boolean = true;
            for (size_t i = 0; i < predicate_count; i++)
            {
                boolean &= preds[i]->operator()(it);
            }
            
            if (boolean)
            {
                if constexpr(generate_stats)
                    ++sel_op_stats.output_cardinality;

                hashtable[*it] = Intermediate_tuple_t({{it.getNumColumns(), it._data}});
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        if constexpr(!generate_stats)
            runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if constexpr(generate_stats)
        {
            sel_op_stats.sample_vector = get_sample_vector<predicate_count>(baseTable_id, preds);
            sel_op_stats.input_cardinality = table.numRows();
            build_op_stats.input_cardinality = sel_op_stats.output_cardinality;
            build_op_stats.output_cardinality = sel_op_stats.output_cardinality;
        }
    }

    json generate_training_data(const std::vector<std::pair<uint64_t,json>> &build_stats = {}) override
    {
        struct op_stats &sel_op_stats = stats[0];
        struct op_stats &build_op_stats = stats[1];

        std::cout << "----- pipeline_sel_probe -----" << std::endl;
        std::cout << "sel_op_stats \n\t input_cardinality: " << sel_op_stats.input_cardinality << "\n\t output_cardinality: " << sel_op_stats.output_cardinality << std::endl;
        std::cout << "build_op_stats \n\t input_cardinality: " << build_op_stats.input_cardinality << std::endl;
        std::cout << "Runtime: " << runtime_us << " us" << std::endl;

        return pipeline::generate_training_data(build_stats);
    }

};

struct pipeline_sel_probe : public pipeline
{    
    virtual uint64_t getType() const override
    {
        return SEL_PROBE;
    }

    template <bool generate_stats, size_t predicate_count>
    void inline run(uint32_t baseTable_id, std::unordered_map<data_type, Intermediate_tuple_t> &hashtable, uint32_t probe_key, PredAbstract **preds, std::vector<Intermediate_tuple_t> &output)
    {
        if constexpr(generate_stats)
        {
            stats.emplace_back(SELECT_BRANCH, baseTable_id);
            for (size_t i = 0; i < predicate_count; ++i)
                stats[0].add_predicate(preds[i]->column, *preds[i]);
            stats.emplace_back(PROBE, baseTable_id);
            stats[1].add_predicate(probe_key, PredEqual(0, probe_key));
        }

        // A bit unsafe, but stats are accessed when generate_stats == false
        struct op_stats &sel_op_stats = stats[0];
        struct op_stats &probe_op_stats = stats[1];

        // std::cout << "Executing pipeline_sel_probe" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto table = tables[baseTable_id];
        for (auto it=table.begin(probe_key); it!=table.end(probe_key); ++it)
        {
            bool boolean = true;
            for (size_t i = 0; i < predicate_count; i++)
            {
                boolean &= preds[i]->operator()(it);
            }
            
            if (boolean)
            {
                if constexpr(generate_stats)
                    ++sel_op_stats.output_cardinality;

                auto right_tuple = hashtable.find(*it);
                if (right_tuple != hashtable.end())
                {
                    if constexpr(generate_stats)
                        ++probe_op_stats.output_cardinality;
                    Intermediate_tuple_t a({{it.getNumColumns(), it._data}, {right_tuple->second.num_cols, right_tuple->second.data}});
                    output.push_back(a);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        if constexpr(!generate_stats)
            runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if constexpr(generate_stats)
        {
            sel_op_stats.sample_vector = get_sample_vector<predicate_count>(baseTable_id, preds);
            sel_op_stats.input_cardinality = table.numRows();
            probe_op_stats.input_cardinality = sel_op_stats.output_cardinality;
        }
    }


    json generate_training_data(const std::vector<std::pair<uint64_t,json>> &build_stats = {}) override
    {
        struct op_stats &sel_op_stats = stats[0];
        struct op_stats &probe_op_stats = stats[1];

        std::cout << "----- pipeline_sel_probe -----" << std::endl;
        std::cout << "sel_op_stats \n\t input_cardinality: " << sel_op_stats.input_cardinality << "\n\t output_cardinality: " << sel_op_stats.output_cardinality << std::endl;
        std::cout << "probe_op_stats \n\t input_cardinality: " << probe_op_stats.input_cardinality << "\n\t output_cardinality: " << probe_op_stats.output_cardinality << std::endl;
        std::cout << "Runtime: " << runtime_us << " us" << std::endl;

        return pipeline::generate_training_data(build_stats);
    }

};

struct pipeline_sel_probe_build : public pipeline
{
    virtual uint64_t getType() const override
    {
        return SEL_PROBE_BUILD;
    }

    template <bool generate_stats, size_t predicate_count>
    void inline run(uint32_t baseTable_id, std::unordered_map<data_type, Intermediate_tuple_t> &hashtable_probe, uint32_t probe_key, std::unordered_map<data_type, Intermediate_tuple_t> &hashtable_build, uint32_t build_key, PredAbstract **preds)
    {
        if constexpr(generate_stats)
        {
            stats.emplace_back(SELECT_BRANCH, baseTable_id);
            for (size_t i = 0; i < predicate_count; ++i)
                stats[0].add_predicate(probe_key, *preds[i]);
            stats.emplace_back(PROBE, baseTable_id);
            stats[1].add_predicate(probe_key, PredEqual(0, probe_key));
            stats.emplace_back(BUILD, baseTable_id);
        }

        // A bit unsafe, but stats are accessed when generate_stats == false
        struct op_stats &sel_op_stats = stats[0];
        struct op_stats &probe_op_stats = stats[1];
        struct op_stats &build_op_stats = stats[2];

        // std::cout << "Executing pipeline_sel_probe_build" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto table = tables[baseTable_id];
        for (auto it=table.begin(build_key); it!=table.end(build_key); ++it)
        {
            bool boolean = true;
            for (size_t i = 0; i < predicate_count; i++)
            {
                boolean &= preds[i]->operator()(it);
            }
            
            if (boolean)
            {
                if constexpr(generate_stats)
                    ++sel_op_stats.output_cardinality;

                auto right_tuple = hashtable_probe.find(it(probe_key));
                if (right_tuple != hashtable_probe.end())
                {
                    if constexpr(generate_stats)
                        ++probe_op_stats.output_cardinality;

                    hashtable_build[*it] = Intermediate_tuple_t({{it.getNumColumns(), it._data}, {right_tuple->second.num_cols, right_tuple->second.data}});
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        if constexpr(!generate_stats)
            runtime_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if constexpr(generate_stats)
        {
            sel_op_stats.sample_vector = get_sample_vector<predicate_count>(baseTable_id, preds);
            sel_op_stats.input_cardinality = table.numRows();
            probe_op_stats.input_cardinality = sel_op_stats.output_cardinality;
            build_op_stats.input_cardinality = probe_op_stats.output_cardinality;
            build_op_stats.output_cardinality = probe_op_stats.output_cardinality;
        }
    }

    json generate_training_data(const std::vector<std::pair<uint64_t,json>> &build_stats = {}) override
    {
        struct op_stats &sel_op_stats = stats[0];
        struct op_stats &probe_op_stats = stats[1];
        struct op_stats &build_op_stats = stats[2];
        std::cout << "----- pipeline_sel_probe_build -----" << std::endl;
        std::cout << "sel_op_stats \n\t input_cardinality: " << sel_op_stats.input_cardinality << "\n\t output_cardinality: " << sel_op_stats.output_cardinality << std::endl;
        std::cout << "probe_op_stats \n\t input_cardinality: " << probe_op_stats.input_cardinality << "\n\t output_cardinality: " << probe_op_stats.output_cardinality << std::endl;
        std::cout << "build_op_stats \n\t input_cardinality: " << build_op_stats.input_cardinality << std::endl;
        std::cout << "Runtime: " << runtime_us << " us" << std::endl;

        return pipeline::generate_training_data(build_stats);
    }
};
