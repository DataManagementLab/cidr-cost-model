#pragma once

#include <exception>
#include <string>

class CidrException : public std::exception
{
public:
    CidrException(const CidrException& other)
    : std::exception(other)
    , filename(other.filename)
    , functionname(other.functionname)
    , lineno(other.lineno)
    , errormsg(other.errormsg)
    , full(other.full)
    { }
    static void throw_exception(const char* file, const char* function, int line, const std::string& msg)
    {
        throw CidrException(file, function, line, msg);
    }

    const std::string&  getFilename( ) const { return filename; }
    const std::string&  getFunction( ) const { return functionname; }
    const std::string&  getErrorMsg( ) const {return errormsg; }
    int                 getLineNumber( ) const { return lineno; }
    virtual const char* what( ) const noexcept override { return full.c_str( ); }
    std::ostream& operator<<(std::ostream& os)
    {
        os << what( );
        return os;
    }
    friend std::ostream& operator<<(std::ostream& os, const CidrException& e)
    {
        os << e.what( );
        return os;
    }
private:
    CidrException( ) = delete;
    CidrException& operator=(const CidrException&) = delete;

    CidrException(const char* file, const char* function, int line, const std::string& msg)
    : std::exception( )
    , filename(file)
    , functionname(function)
    , lineno(line)
    , errormsg(msg)
    {
        std::stringstream ss;
        ss  << function << ": " << msg << " (" << file << ':' << line << ')';
        full = ss.str( );
    }

    std::string filename;
    std::string functionname;
    int lineno;
    std::string errormsg;
    std::string full;
};

#define THROW_CIDR_EXCEPTION(msg_stream)                                                           \
  while (true)                                                                                     \
  {                                                                                                \
      std::stringstream __ss_stream;                                                               \
      __ss_stream << msg_stream;                                                                   \
      CidrException::throw_exception(__FILE__, __PRETTY_FUNCTION__, __LINE__, __ss_stream.str( )); \
  }

