
#pragma once
#include <vector>
#include <memory>
#include "../execution_engine/predicate.hpp"
#include "../storage/storage.hpp"
#include "../storage/column_storage.hpp"
#include "../storage/row_storage.hpp"


//! Scans over storage with branching - returns position list
/**
 *  \tparam Iterator iterator for column on table, to be deduced
 */
template<typename Iterator, typename Predicate_t>
std::unique_ptr<std::vector<uint32_t> > selection_op_branch(const Iterator& begin, const Iterator& end, const Predicate_t &pred)
{
    std::unique_ptr<std::vector<uint32_t> > posList = std::unique_ptr<std::vector<uint32_t> >(new std::vector<uint32_t>());
    posList->reserve(end.getCurrentRowId( ));

    for (Iterator it=begin; it!=end; ++it)
        if (pred(it))
            posList->push_back(it.getCurrentRowId( ));

    return posList;
}


//Scans over storage without branching - returns position list
template<typename T, typename Predicate_t>
std::vector<uint32_t>* selection_op_branchless(const Storage<T> &storage, const Predicate_t &pred)
{
    std::vector<uint32_t> *posList = new std::vector<uint32_t>();
    posList->resize(storage.numRows());
    auto iter = storage.begin();
    auto end = storage.end();

    uint32_t posListPos = 0;
    while (iter != end)
    {
        posList->operator[](posListPos) = iter.pos;
        posListPos += pred(iter) ? 1 : 0;
        iter++;
    }

    return posList;
}

//===================================



// //Access storage on pass position list with branching - returns position list
// template<typename T, typename Predicate_t, bool branching = true>
// std::vector<uint32_t>* selection_op(const Storage<T> &storage, std::vector<uint32_t>& posList, const Predicate_t &pred)
// {
//     std::vector<uint32_t> *retPosList = new std::vector<uint32_t>();
//     retPosList->reserve(numRows);
//     auto iter = storage.begin();
//     for (uint32_t pos : posList)
//     {
//         if (pred(&iter[pos]))
//         {
//             retPosList->push_back(pos);
//         }
//     }

//     return retPosList;
// }


// //Scans over storage without branching - returns position list
// template<typename T, typename Predicate_t>
// std::vector<uint32_t>* selection_op<Predicate_t, false>(const Storage<T> &storage, std::vector<uint32_t>& posList, const Predicate_t &pred)
// {
//     std::vector<uint32_t> *posList = new std::vector<uint32_t>();
//     posList->resize(storage.numRows());
//     auto iter = storage.begin();
//     auto end = storage.end();

//     uint32_t posListPos = 0;
//     for (uint32_t pos : posList)
//     {
//         posList->operator[posListPos] = pos;
//         posListPos += pred(&iter[pos]) ? 1 : 0;
//     }

//     return posList;
// }

