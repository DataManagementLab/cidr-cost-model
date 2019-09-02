#include <stdint.h>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <numeric>
#include <random>

#include "pipelines.h"

#include "../storage/table.hpp"
#include "../storage/filehelper.h"
#include "../execution_engine/predicate.hpp"
#include "../execution_engine/dump_types.hpp"
#include "../execution_engine/dump_json.hpp"

#include <nlohmann/json.hpp>
using nlohmann::json;

// --------------------------- GLOBAL STUFF ---------------------------


//pairs of #attributes-#rows
std::vector<std::pair<uint32_t, uint32_t>> table_confs = {

    // {1,1250000},{8,1250000/8},{16,1250000/16}, //5mb
    // {1,2500000},{8,2500000/8},{16,2500000/16}, //10mb
    // {1,5000000},{8,5000000/8},{16,5000000/16}, //20mb
    // {1,10000000},{8,10000000/8},{16,10000000/16}, //40mb
    // {1,20000000},{8,20000000/8},{16,20000000/16}, //80mb
    // {1,40000000},{8,40000000/8},{16,40000000/16}, //160mb
    // {1,80000000},{8,80000000/8},{16,80000000/16}, //320mb
    // {1,160000000},{8,160000000/8},{16,160000000/16}, //640mb
    // {1,320000000},{8,320000000/8},{16,320000000/16}, //1280mb
    {1,64000000},{8,64000000/8},{16,64000000/16}, //256mb
    {1,128000000},{8,128000000/8},{16,128000000/16}, //512mb
    {1,256000000},{8,256000000/8},{16,256000000/16}, //1024mb
};


uint64_t getTotalColumnCount()
{
    uint64_t res = 0;
    for(const auto& t : table_confs)
    {
        res += t.first;
    }

    return res;
}

void to_json(json &j, const std::pair<uint64_t,PredAbstract> &p)
{
    //j = json{{"column_id", p.first}};
    j = json{{"column_id", p.first},{"pred_type", p.second.getPredicateType()}, {"literal_normalized", normalize_column_value(p.first, p.second.getCompareValue())}};
}


// --------------------------- QUERY PLANS ---------------------------
template <size_t predicate_count>
json query_plan_simple_sel(uint32_t table_id, PredAbstract **preds, std::ofstream *csv_plot_file = nullptr)
{
    mem_pool_offset = 0;
    pipeline_sel p1;
    {   //REAL EXECUTION
        std::vector<Intermediate_tuple_t> result;
        result.reserve(tables[table_id].numRows());
        p1.run<false, predicate_count>(table_id, preds, result);
    }
    {   //STAT GENERATION
        std::vector<Intermediate_tuple_t> result;
        result.reserve(tables[table_id].numRows());
        p1.run<true, predicate_count>(table_id, preds, result);
    }
    std::cout << "********************* query_plan_simple_sel *********************" << std::endl;
    auto j1 = p1.generate_training_data();

    //csv for plotting
    struct op_stats &sel_op_stats = p1.stats[0];
    if (csv_plot_file != nullptr)
    {
        (*csv_plot_file) << tables[table_id].numRows() << "," << tables[table_id].numRows()*tables[table_id].numColumns()*sizeof(data_type)/1000 << "," << 1.0*sel_op_stats.output_cardinality/sel_op_stats.input_cardinality << "," << tables[table_id].numColumns() << "," << predicate_count << "," << p1.runtime_us << ",branch\n";
    }

    return query_stats{0,p1.runtime_us, j1, tables[table_id].numRows()*tables[table_id].numColumns()*sizeof(data_type)/1000, 1.0*sel_op_stats.output_cardinality/sel_op_stats.input_cardinality, tables[table_id].numColumns() ,predicate_count};

}


template <size_t predicate_count>
json query_plan_1(uint32_t tableA_id, uint32_t tableB_id, uint32_t probe_key, uint32_t build_key, PredAbstract **preds1, PredAbstract **preds2, std::ofstream *csv_probe_pipeline_file = nullptr, std::ofstream *csv_build_pipeline_file = nullptr)
{
    mem_pool_offset = 0;
    pipeline_sel_build p1;
    pipeline_sel_probe p2;

    {   //REAL EXECUTION
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable;
        std::vector<Intermediate_tuple_t> result;
        result.reserve(std::max(tables[tableA_id].numRows(), tables[tableB_id].numRows()));
        hashtable.reserve(std::max(tables[tableA_id].numRows(), tables[tableB_id].numRows()));
        p1.run<false, predicate_count>(tableA_id, hashtable, build_key, preds1);
        p2.run<false, predicate_count>(tableB_id, hashtable, probe_key, preds2, result);
    }
    std::vector<Intermediate_tuple_t> result;
    std::unordered_map<data_type, Intermediate_tuple_t> hashtable;
    {   //STAT GENERATION
        result.reserve(std::max(tables[tableA_id].numRows(), tables[tableB_id].numRows()));
        hashtable.reserve(std::max(tables[tableA_id].numRows(), tables[tableB_id].numRows()));
        p1.run<true, predicate_count>(tableA_id, hashtable, build_key, preds1);
        p2.run<true, predicate_count>(tableB_id, hashtable, probe_key, preds2, result);
    }
    std::cout << "********************* query_plan_1 *********************" << std::endl;
    auto j1 = p1.generate_training_data();
    //! TODO fixme, need unique (absolute) column id in whole DB
    uint32_t build_key_abs = to_abs_column_id(tableA_id,build_key);
    auto j2 = p2.generate_training_data({{build_key_abs, j1}});
    auto total_runtime_us = p1.runtime_us + p2.runtime_us;
    std::cout << "TOTAL QUERY RUNTIME: " << total_runtime_us << std::endl;

    //CSV for probe pipeline
    if (csv_probe_pipeline_file != nullptr)
    {
        (*csv_probe_pipeline_file) << tables[tableA_id].numRows() << "," << tables[tableA_id].numRows()*tables[tableA_id].numColumns()*sizeof(data_type)/1000 << "," << 1.0*hashtable.size()/tables[tableA_id].numRows() << "," << tables[tableA_id].numColumns() << "," << predicate_count << "," << p1.runtime_us << ",branch\n";
    }
    //CSV for build pipeline
    if (csv_build_pipeline_file != nullptr)
    {
        (*csv_build_pipeline_file) << tables[tableB_id].numRows() << "," << tables[tableB_id].numRows()*tables[tableB_id].numColumns()*sizeof(data_type)/1000 << "," << 1.0*result.size()/tables[tableB_id].numRows() << "," << tables[tableB_id].numColumns() << "," << predicate_count << "," << p2.runtime_us << ",branch\n";
    }

    return query_stats{1,total_runtime_us, j2, tables[tableA_id].numRows()*tables[tableA_id].numColumns()*sizeof(data_type)/1000, 1.0*result.size()/tables[tableA_id].numRows(), tables[tableA_id].numColumns(), predicate_count};
}

template <size_t predicate_count>
json query_plan_2(uint32_t tableA_id, uint32_t tableB_id, uint32_t tableC_id, uint32_t probe_key, uint32_t build_key, PredAbstract **preds1, PredAbstract **preds2, PredAbstract **preds3)
{
    mem_pool_offset = 0;
    pipeline_sel_build p1;
    pipeline_sel_probe_build p2;
    pipeline_sel_probe p3;
    {   //REAL EXECUTION
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable1;
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable2;
        std::vector<Intermediate_tuple_t> result;
        hashtable1.reserve(std::max({tables[tableA_id].numRows(), tables[tableB_id].numRows(), tables[tableC_id].numRows()}));
        hashtable2.reserve(std::max({tables[tableA_id].numRows(), tables[tableB_id].numRows(), tables[tableC_id].numRows()}));
        result.reserve(std::max({tables[tableA_id].numRows(), tables[tableB_id].numRows(), tables[tableC_id].numRows()}));
        p1.run<false, predicate_count>(tableA_id, hashtable1, build_key, preds1);
        p2.run<false, predicate_count>(tableB_id, hashtable1, probe_key, hashtable2, build_key, preds2);
        p3.run<false, predicate_count>(tableC_id, hashtable2, probe_key, preds3, result);
    }
    std::vector<Intermediate_tuple_t> result;
    {   //STAT GENERATION
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable1;
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable2;
        hashtable1.reserve(std::max({tables[tableA_id].numRows(), tables[tableB_id].numRows(), tables[tableC_id].numRows()}));
        hashtable2.reserve(std::max({tables[tableA_id].numRows(), tables[tableB_id].numRows(), tables[tableC_id].numRows()}));
        result.reserve(std::max({tables[tableA_id].numRows(), tables[tableB_id].numRows(), tables[tableC_id].numRows()}));

        p1.run<true, predicate_count>(tableA_id, hashtable1, build_key, preds1);
        p2.run<true, predicate_count>(tableB_id, hashtable1, probe_key, hashtable2, build_key, preds2);
        p3.run<true, predicate_count>(tableC_id, hashtable2, probe_key, preds3, result);
    }

    //! TODO \todo{convert build_key to absolute column}
    uint32_t build_key_abs1 = to_abs_column_id(tableA_id,build_key);
    uint32_t build_key_abs2 = to_abs_column_id(tableB_id,build_key);
    std::cout << "********************* query_plan_2 *********************" << std::endl;
    auto j1 = p1.generate_training_data();
    auto j2 = p2.generate_training_data({{build_key_abs1, j1}});
    auto j3 = p3.generate_training_data({{build_key_abs2, j2}});
    auto total_runtime_us = p1.runtime_us + p2.runtime_us + p3.runtime_us;
    std::cout << "TOTAL QUERY RUNTIME: " << total_runtime_us << std::endl;
    
    return query_stats{2,total_runtime_us, j3, tables[tableA_id].numRows()*tables[tableA_id].numColumns()*sizeof(data_type)/1000, 1.0*result.size()/tables[tableA_id].numRows(), tables[tableA_id].numColumns(), predicate_count};
}


template <size_t predicate_count>
json query_plan_3(uint32_t tableA_id, uint32_t tableB_id, uint32_t tableC_id, uint32_t tableD_id, uint32_t probe_key, uint32_t build_key, PredAbstract **preds1, PredAbstract **preds2, PredAbstract **preds3, PredAbstract **preds4)
{
    mem_pool_offset = 0;
    pipeline_sel_build p1;
    pipeline_sel_probe_build p2;
    pipeline_sel_probe_build p3;
    pipeline_sel_probe p4;
    {   //REAL EXECUTION
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable1;
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable2;
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable3;
        std::vector<Intermediate_tuple_t> result;

        p1.run<false, predicate_count>(tableA_id, hashtable1, build_key, preds1);
        p2.run<false, predicate_count>(tableB_id, hashtable1, probe_key, hashtable2, build_key, preds2);
        p3.run<false, predicate_count>(tableC_id, hashtable2, probe_key, hashtable3, build_key, preds3);
        p4.run<false, predicate_count>(tableD_id, hashtable3, probe_key, preds4, result);
    }
    std::vector<Intermediate_tuple_t> result;
    {   //STAT GENERATION
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable1;
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable2;
        std::unordered_map<data_type, Intermediate_tuple_t> hashtable3;

        p1.run<true, predicate_count>(tableA_id, hashtable1, build_key, preds1);
        p2.run<true, predicate_count>(tableB_id, hashtable1, probe_key, hashtable2, build_key, preds2);
        p3.run<true, predicate_count>(tableC_id, hashtable2, probe_key, hashtable3, build_key, preds3);
        p4.run<true, predicate_count>(tableD_id, hashtable3, probe_key, preds4, result);
    }
    std::cout << "********************* query_plan_3 *********************" << std::endl;
    uint32_t build_key_abs1 = to_abs_column_id(tableA_id,build_key);
    uint32_t build_key_abs2 = to_abs_column_id(tableB_id,build_key);
    uint32_t build_key_abs3 = to_abs_column_id(tableC_id,build_key);
    auto j1 = p1.generate_training_data();
    auto j2 = p2.generate_training_data({{build_key_abs1, j1}});
    auto j3 = p3.generate_training_data({{build_key_abs2, j2}});
    auto j4 = p4.generate_training_data({{build_key_abs3, j3}});
    auto total_runtime_us = p1.runtime_us + p2.runtime_us + p3.runtime_us + p4.runtime_us;
    std::cout << "TOTAL QUERY RUNTIME: " << total_runtime_us << std::endl;

    return query_stats{3,total_runtime_us, j4, tables[tableA_id].numRows()*tables[tableA_id].numColumns()*sizeof(data_type)/1000, 1.0*result.size()/tables[tableA_id].numRows(), tables[tableA_id].numColumns(), predicate_count};
}


void generate_tables()
{
    std::cout << "Generating tables..." << std::endl;

    assert(table_confs.size() <= num_tables);

    for (size_t table_id = 0; table_id < table_confs.size(); table_id++)
    {
        const size_t num_columns = table_confs[table_id].first;
        const size_t num_rows = table_confs[table_id].second;

        Schema schema;
        std::vector<int> col_ids(num_columns); 
        std::iota(col_ids.begin(), col_ids.end(), 0);
        schema.add_column(col_ids); //Create row-store

        tables[table_id] = Table<data_type>(schema);

        std::vector<data_type> values(num_rows);
        for (data_type row=0; row<num_rows; ++row)
            values[row] = row;
            
        std::random_shuffle(values.begin(), values.end());
        
        data_type *data = new data_type[num_rows*num_columns];
        for (data_type row=0; row<num_rows; ++row)
            for (data_type col=0; col<num_columns; ++col)
            {
                data[row*num_columns + col] = values[row];
                // switch (col % 3)
                // {
                // case 0: //Immitate PK column
                //     data[row*num_columns + col] = row;
                //     break;
                // case 1: //Immitate FK-like column with fewer distinct values
                //     data[row*num_columns + col] = rand() % 500;
                //     break;
                // case 2: //Random values
                //     data[row*num_columns + col] = rand();
                //     break;
                // default:
                //     break;
                // }
            }
        
        tables[table_id].loadFromMemory(data, num_rows);
        delete[] data;
        std::cout << "Finished generating table id: " << table_id << std::endl;
    }

}

void simple_selection_experiment()
{
    std::string filename = "../simple_sel_pipeline_new_tablesizes.csv";
    std::string json_filename = "../simple_sel_pipeline_new_tablesizes.json";

    if (!Filehelper::fileExists(filename))
    {
      std::ofstream csv_plot_file(filename, std::ios_base::app | std::ios_base::out);
      csv_plot_file << "tablerows,tablesizekb,selectivity,tuplewidth,attrInPred,runtime,operator\n";
      csv_plot_file.flush();
    }
    std::vector<json> queries_json;

    for (size_t i = 0; i < table_confs.size(); i++)
    {
        std::ofstream csv_plot_file(filename, std::ios_base::app | std::ios_base::out);

        const size_t num_columns = table_confs[i].first;
        const size_t num_rows = table_confs[i].second;
        std::cout << "Generating table with " << num_columns << " columns and " << num_rows << " rows." << std::endl;

        Schema schema;
        std::vector<int> col_ids(num_columns); 
        std::iota(col_ids.begin(), col_ids.end(), 0);
        schema.add_column(col_ids); //Create row-store

        std::vector<data_type> values(num_rows);
        for (data_type row=0; row<num_rows; ++row)
            values[row] = row;

        std::random_shuffle(values.begin(), values.end());

        tables[i] = Table<data_type> (schema);
        data_type *data = new data_type[num_rows*num_columns];
        for (data_type row=0; row<num_rows; ++row)
            for (data_type col=0; col<num_columns; ++col)
                data[row*num_columns + col] = values[row];

        values.clear();

        tables[i].loadFromMemory(data, num_rows);

        for (size_t sel = 0; sel <= 100; sel+=5)
        {
            PredAbstract **preds = new PredAbstract*[16];
            for (size_t i = 0; i < 16; i++)
            {
                preds[i] = new PredSel(i, sel, num_rows);
            }
            
            queries_json.push_back(query_plan_simple_sel<1>(i, preds, &csv_plot_file));
            if (num_columns < 2) continue;
            queries_json.push_back(query_plan_simple_sel<2>(i, preds, &csv_plot_file));
            if (num_columns < 4) continue;
            queries_json.push_back(query_plan_simple_sel<4>(i, preds, &csv_plot_file));
            if (num_columns < 8) continue;
            queries_json.push_back(query_plan_simple_sel<8>(i, preds, &csv_plot_file));
            if (num_columns < 16) continue;
            queries_json.push_back(query_plan_simple_sel<16>(i, preds, &csv_plot_file));
        }
#ifdef DEBUG
        // std::cout << j << std::endl;
#endif
        
        delete[] data;
    
    }

    json j = {{"queries",queries_json}, {"meta", meta_stats{table_confs.size(),getTotalColumnCount(),get_table_to_col_map(table_confs.size())}}};

    std::ofstream json_file(json_filename, std::ios_base::trunc | std::ios_base::out);
    json_file << j << std::endl;
    json_file.close();

}




int main(int, char const **)
{    
    srand(1);

    simple_selection_experiment();
    
    // std::cout << "Finished simple_selection_experiment" << std::endl;

    // generate_tables();
    // std::string json_filename = "../query_plan_stats.json";
    // std::string filename_build = "../build_pipeline.csv";
    // std::string filename_probe = "../probe_pipeline.csv";
    // if (!Filehelper::fileExists(filename_build))
    // {
    //   std::ofstream csv_probe_file(filename_build, std::ios_base::app | std::ios_base::out);
    //   csv_probe_file << "tablerows,tablesizekb,selectivity,tuplewidth,attrInPred,runtime,operator\n";
    //   csv_probe_file.flush();
    // }

    // if (!Filehelper::fileExists(filename_probe))
    // {
    //   std::ofstream csv_build_file(filename_probe, std::ios_base::app | std::ios_base::out);
    //   csv_build_file << "tablerows,tablesizekb,selectivity,tuplewidth,attrInPred,runtime,operator\n";
    //   csv_build_file.flush();
    // }
    // std::vector<json> queries_json;
    
    // for (size_t table_id = 0; table_id < table_confs.size(); table_id++)
    // {
    //     std::ofstream csv_probe_file(filename_build, std::ios_base::app | std::ios_base::out);
    //     std::ofstream csv_build_file(filename_probe, std::ios_base::app | std::ios_base::out);

    //     for (size_t sel = 0; sel <= 100; sel+=5)
    //     {
    //         PredAbstract **preds = new PredAbstract*[16];
    //         for (size_t i = 0; i < 16; i++)
    //         {
    //             preds[i] = new PredSel(i, sel, tables[table_id].numRows());
    //         }
            
    //         queries_json.push_back(query_plan_1<1>(table_id, table_id, 0, 0, preds, preds, &csv_probe_file, &csv_build_file));
    //         if (tables[table_id].numColumns() < 2) continue;
    //         queries_json.push_back(query_plan_1<2>(table_id, table_id, 0, 0, preds, preds, &csv_probe_file, &csv_build_file));
    //         if (tables[table_id].numColumns() < 4) continue;
    //         queries_json.push_back(query_plan_1<4>(table_id, table_id, 0, 0, preds, preds, &csv_probe_file, &csv_build_file));
    //         if (tables[table_id].numColumns() < 8) continue;
    //         queries_json.push_back(query_plan_1<8>(table_id, table_id, 0, 0, preds, preds, &csv_probe_file, &csv_build_file));
    //         if (tables[table_id].numColumns() < 16) continue;
    //         queries_json.push_back(query_plan_1<16>(table_id, table_id, 0, 0, preds, preds, &csv_probe_file, &csv_build_file));
    //     }

    // }

    // json j = {{"queries",queries_json}, {"meta", meta_stats{table_confs.size(),getTotalColumnCount(),get_table_to_col_map(table_confs.size())}}};

    // std::ofstream json_file(json_filename, std::ios_base::trunc | std::ios_base::out);
    // json_file << j << std::endl;
    // json_file.close();
    return 0;
}
