#pragma once

#include <iostream>
#include <string>

class FileStreamGuard
{
public:
    FileStreamGuard(const std::string& filename)
    : stream(filename)
    { }
    virtual ~FileStreamGuard( )
    {
        if (stream.is_open( ))
            stream.close( );
    }

    bool good( )
    {
        return stream.good( );
    }
    std::ifstream& get( )
    {
        return stream;
    }

private:
    std::ifstream stream;
};

