#pragma once
#include <stdint.h>
#include <iostream>

#include "dump_types.hpp"
// Interface for predicate
/*
*
*   tuple_size: Assumed size of the tuple addressed by iterator
*   tuple: Iterator to tuple in storage object for assumed tuple size!
*/
template<typename Iterator>
struct Predicate
{
protected:
    pred_type pt = pred_type::INVALID_PRED;
    size_t value = 0;

public:
    size_t column = 0;
    
    Predicate() = default;

    Predicate(pred_type _pt, size_t _value, size_t _column = 0): pt(_pt), value(_value), column(_column)
    {}

    virtual bool operator()( [[maybe_unused]] const Iterator& tuple) const
    {
        return false;
    }
    virtual pred_type getPredicateType() const noexcept
    {
        return pt;
    }

    virtual double getCompareValue() const noexcept
    {
        return value;
    }

};


// For testing
template<typename Iterator>
struct PredicateTrue : public Predicate<Iterator>
{
    bool inline operator()(const Iterator&) const override
    {
        // assert(assumed_width == attribute.dataSize);
        return true;
    }
};


// For testing
template<typename Iterator>
struct PredicateHalf : public Predicate<Iterator>
{
    using Parent = Predicate<Iterator>;
    PredicateHalf(size_t column): Parent(pred_type::HALF, 0, column)
    {
    }
    
    bool inline operator()(const Iterator& tuple) const override
    {
        // assert(assumed_width == attribute.dataSize);
        // Tiemo: was value(Parrent::column), bug?
        return (tuple(Parent::column) % 2 == 0);
    }
};

// For testing with specific selectivity
template<typename Iterator>
struct PredicateSelectivity : public Predicate<Iterator>
{
    using Parent = Predicate<Iterator>;
    PredicateSelectivity(size_t column, size_t selectivity, size_t num_rows): Parent(pred_type::LESS, 0, column)
    {
        Parent::value = num_rows / 100 * selectivity;
    }
    
    bool inline operator()(const Iterator& tuple) const override
    {
        return (tuple(Parent::column) < Parent::value);
    }
};




template<typename Iterator>
struct PredicateGreaterThan : public Predicate<Iterator>
{
    using Parent = Predicate<Iterator>;

    PredicateGreaterThan(size_t value, size_t column): Parent(pred_type::GREATER, value, column)
    {
    }
    
    bool inline operator()(const Iterator& tuple) const override
    {
        // assert(assumed_width == attribute.dataSize);
        return (tuple(Parent::column) > Parent::value);
    }
};

template<typename Iterator>
struct PredicateLessThan : public Predicate<Iterator>
{
    using Parent = Predicate<Iterator>;

    PredicateLessThan(size_t value, size_t column): Parent(pred_type::LESS, value, column)
    {}
    
    bool inline operator()(const Iterator& tuple) const override
    {
        // assert(assumed_width == attribute.dataSize);
        return (tuple(Parent::column) < Parent::value);
    }
};

template<typename Iterator>
struct PredicateEqual : public Predicate<Iterator>
{
    using Parent = Predicate<Iterator>;

    PredicateEqual(size_t value, size_t column): Parent(pred_type::EQUAL, value, column)
    {}
    
    bool inline operator()(const Iterator& tuple) const override
    {
        // assert(assumed_width == attribute.dataSize);
        return (tuple(Parent::column) == Parent::value);
    }
};

// AND predicate for variable attributes in the same tuple
template<typename Iterator>
struct PredicateAnd : public Predicate<Iterator>
{
    bool inline operator()(const Iterator& tuple) const
    {
        // assert(assumed_width == attribute.dataSize);
        const uint64_t tuple_size = tuple.dataSize;//Iterator::DataTypeSize;

        bool res = true;

        auto* attribute = &(*tuple);
        for(uint64_t i=0; i < tuple_size; ++i)
        {
            res &= *attribute;
            ++attribute;
        }
        return res;
    }
};

// Assumed interface of operator
/*
*   Template parameters:
*       Iterator: Iterator type of stroage object
*       Predicate: Type of predicate to evaluate
*   storage_object: Iterator to storage object for access to tuples 
*   p:  predicate to evaluate on tuples of storage object
*
*   Output: PositionList or materialized result
 */
template<typename Iterator, typename Predicate, typename PositionList>
PositionList select(const Iterator& storage_object, const Predicate& p)
{
    // throw std::runtime_error("Do not use, only an example of the assumed interface!");
    for(const auto& tuple: storage_object)
    {
        bool match = p(tuple);
    }
}
