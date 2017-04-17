#ifndef PYROL_HPP
#define PYROL_HPP

// Python Includes
#include "Python.h"

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_AsString PyUnicode_AsString
#else 
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
#endif

#define  C_TEXT(text) ((char*)std::string(text).c_str())

// C++ Includes
#include <algorithm>
#include <ostream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>


// ROL Includes
#include "ROL_ElementwiseVector.hpp"
#include "ROL_RandomVector.hpp"
#include "ROL_StdVector.hpp"

#endif // PYROL_HPP
