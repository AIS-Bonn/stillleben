// Base exception class
// Author: Max Schwarz <max.schwarz@ais.uni-bonn.de>

#ifndef STILLLEBEN_EXCEPTION_H
#define STILLLEBEN_EXCEPTION_H

#include <stdexcept>

namespace sl
{

class Exception : public std::runtime_error
{
public:
    Exception(const char* msg);
};

}

#endif
