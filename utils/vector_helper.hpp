
#pragma once

#include <iostream>

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
    out << '{';
    if (v.size( ) > 0)
    {
        out << v[0];
        for (typename vector<T>::size_type i=1; i<v.size( ); ++i)
          out << ", " << v[i];
    }
    out << '}';
    return out;
}

