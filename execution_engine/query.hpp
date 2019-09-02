#pragma once

template<typename Table_t>
struct Query
{
    virtual void execute(Table_t &table) = 0;
};