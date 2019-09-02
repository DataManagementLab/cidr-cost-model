#include <stdint.h>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>

static bool fileExists(const std::string& file) {
    struct stat buffer;
    return (stat(file.c_str(), &buffer) == 0);
}

// template <typename T>
// class OutputArray
// {
// private:
//     size_t array_size;
//     T* arrays;
//     std::vector<T*> array_start_ptrs;
//     size_t current_array_idx = 0;
//     int last_index = 0;
//     T* cur_array;
// public:
//     OutputArray(size_t ar_size) : array_size(ar_size)
//     {
//         cur_array = new T[array_size];
//         // array_start_ptrs.reserve(1000);
//         // auto ptr = new T[array_size];
//         // array_start_ptrs.push_back(ptr);
//     };

//     T& operator[] (const int index)
//     {
//         return cur_array[index];

//         // if (index != last_index && index % array_size == 0) // allocate a new array
//         // {
//         //     auto ptr = new T[array_size];
//         //     array_start_ptrs.push_back(ptr);
//         //     current_array_idx++;
//         //     last_index = index;
//         // }
//         // return array_start_ptrs[current_array_idx][index % array_size];
//     };
// };

struct Tuple1Attr
{
    uint64_t attr[1];
};
struct Tuple8Attr
{
    uint64_t attr[8];
};
struct Tuple16Attr
{
    uint64_t attr[16];
};

template <typename Tuple_t, size_t attr_preds>
void branch_selection(Tuple_t *table_ptr, size_t table_size, Tuple_t* output, float selectivity)
{
    size_t ptr = 0;
    auto sel_thres = (uint64_t)(selectivity*table_size);
    for (size_t i = 0; i < table_size; i++)
    {
        bool pred = true;
        for (size_t j = 0; j < attr_preds; j++)
        {
            pred &= table_ptr[i].attr[j] < sel_thres;
        }
        
        if (pred)
        {
            output[ptr] = table_ptr[i];
            ptr++;
        }
    }
    // std::cout << "Written tuples: " << ptr << std::endl;
};

template <typename Tuple_t, size_t attr_preds>
void branchless_selection(Tuple_t *table_ptr, size_t table_size, Tuple_t* output, float selectivity)
{
    auto sel_thres = (uint64_t)(selectivity*table_size);
    size_t ptr = 0;
    for (size_t i = 0; i < table_size; i++)
    {
        output[ptr] = table_ptr[i];

        bool pred = true;
        for (size_t j = 0; j < attr_preds; j++)
        {
            pred &= table_ptr[i].attr[j] < sel_thres;
        }
        ptr += pred ? 1 : 0; 
    }
    // std::cout << "Written tuples: " << ptr << std::endl;
};


template <typename Tuple_t, size_t attr_preds>
void benchmark(size_t table_size, std::ofstream &result_file)
{

    Tuple_t* table = new Tuple_t[table_size];
    for (size_t i = 0; i < table_size; i++)
    {
        for (size_t j = 0; j < attr_preds; j++)
        {
            table[i].attr[j] = i;
        }
    }

    std::random_shuffle(table, table + table_size);

    for (size_t sel = 0; sel <= 100; sel+=5)
    {
        Tuple_t *output = new Tuple_t[table_size];
        memset(output, 0, table_size * sizeof(Tuple_t));
        auto start = std::chrono::high_resolution_clock::now();
        branchless_selection<Tuple_t, attr_preds>(table, table_size, output, (float)sel/100);
        auto end = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        result_file << table_size << "," << sel << "," << sizeof(Tuple_t) << "," << attr_preds << "," << runtime << ",branchless\n";
        std::cout << "Tuple size: " << sizeof(Tuple_t) << " Attr in pred: " << attr_preds << " Selectivity: " << sel << "% Runtime: " << runtime/1000 << "ms - BRANCHLESS \n";
        delete[] output;
    }
    for (size_t sel = 0; sel <= 100; sel+=5)
    {
        Tuple_t *output = new Tuple_t[table_size];
        memset(output, 0, table_size * sizeof(Tuple_t));
        auto start = std::chrono::high_resolution_clock::now();
        branch_selection<Tuple_t, attr_preds>(table, table_size, output, (float)sel/100);
        auto end = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        result_file << table_size << "," << sel << "," << sizeof(Tuple_t) << "," << attr_preds << "," << runtime << ",branch\n";
        std::cout << "Tuple size: " << sizeof(Tuple_t) << " Attr in pred: " << attr_preds << " Selectivity: " << sel << "% Runtime: " << runtime/1000 << "ms - BRANCH \n";
        delete[] output;
    }

    delete[] table;
}

int main(int argc, char const *argv[])
{
    std::string filename = "physical_cost_exp1.csv";

    if (!fileExists(filename))
    {
      std::ofstream result_file(filename, std::ios_base::app | std::ios_base::out);
      result_file << "tablesize,selectivity,tuplewidth,attrInPred,runtime,operator\n";
      result_file.flush();
    }

    std::ofstream result_file(filename, std::ios_base::app | std::ios_base::out);
    
    auto table_size = 20000000;

    // 1 ATTRIBUTE - 1 PREDICATE
    benchmark<Tuple1Attr, 1>(table_size, result_file);

    // 8 ATTRIBUTEs - 1 PREDICATE
    benchmark<Tuple8Attr, 1>(table_size, result_file);

    // 16 ATTRIBUTEs - 1 PREDICATE
    benchmark<Tuple16Attr, 1>(table_size, result_file);

    // 8 ATTRIBUTEs - 8 PREDICATEs
    benchmark<Tuple8Attr, 8>(table_size, result_file);

    // 16 ATTRIBUTEs - 8 PREDICATEs
    benchmark<Tuple16Attr, 8>(table_size, result_file);

    // 16 ATTRIBUTEs - 16 PREDICATEs
    benchmark<Tuple16Attr, 16>(table_size, result_file);


    return 0;
}
