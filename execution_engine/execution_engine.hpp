#pragma once

#include <array>
#include <cassert>

#include "schema.h"
#include "table.hpp"

//! Execution engine integrating grey-box storage and operators
/**
 *  \tparam EngineConfig Struct with configuration of execution engine
 */
template<typename EngineConfig>
class ExecutionEngine
{
    public:

    //! Type of tables to process, currently assuming single type
    using Table_t = typename EngineConfig::Table_t;
    //! Number of tables defining size of table array
    static const uint64_t numTables = EngineConfig::numTables;

    //! Constructor to load table schemata
    /**
    *   \param schemaFiles  List of files which each describe a schema of a table
    *
    *   \pre Size of \p schemaFiles must equal \a numTables
    * 
    *   \warn{Copy constructor}
    */
    ExecutionEngine(const std::vector<std::string> &schemaFiles)
    {
        // Checking precondition
        assert( schemaFiles.size() == tables.size());

        for(uint64_t i=0; i < tables.size(); ++i)
        {
            Schema s;
            s.loadFromCsv(schemaFiles.at(i));
            tables.at(i) = Table_t(s);
        }
    }

    //! Constructor to load table schemata
    /**
    *   \param schemaFiles  List of files each describing a schema of a table
    *   \param dataFiles    List of files containing data for table at same index as schemaFiles
    * 
    *   \pre Size of \p schemaFiles must equal \a numTables
    *   \pre Index of file in \p dataFiles indicates index of table in \a tables 
    */
    ExecutionEngine(const std::vector<std::string> &schemaFiles
    , const std::vector<std::string> &dataFiles) : ExecutionEngine(schemaFiles)
    {

#ifdef DEBUG
        if(dataFiles.size() < schemaFiles.size())
        {
            std::cout << "WARN: Loading data for fewer tables than exsisting!\n";
        }
#endif

        for(uint64_t i = 0; i < dataFiles.size(); ++i)
        {
            tables.at(i).loadFromCsv(dataFiles.at(i));
        }
    }

    Table_t& getTable(uint64_t i)
    {
        return tables[i];
    }


    template<typename Query_t>
    void execute(uint64_t table)
#ifdef DEBUG
        noexcept
#endif
    {

        Query_t q;
        q.execute(tables[table]);
    }


    private:

    std::array<Table_t, EngineConfig::numTables> tables;
};
