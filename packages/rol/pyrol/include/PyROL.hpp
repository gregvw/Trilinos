#ifndef PYROL_HPP
#define PYROL_HPP

#ifdef ENABLE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Python Includes
#include "Python.h"
#ifdef ENABLE_NUMPY
#include "numpy/arrayobject.h"
#endif


#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString
#define PyString_AsString PyBytes_AS_STRING
#else
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
#endif

#define  C_TEXT(text) ((char*)std::string(text).c_str())

// Apparently this is not getting set from CMake, but ENABLE_NUMPY does
//#ifndef PYROL_DEBUG_MODE
//#define PYROL_DEBUG_MODE 1
//#endif


// C++ Includes
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>


// ROL Includes
#include "ROL_ElementwiseVector.hpp"
#include "ROL_RandomVector.hpp"
#include "ROL_StdVector.hpp"

#endif // PYROL_HPP
