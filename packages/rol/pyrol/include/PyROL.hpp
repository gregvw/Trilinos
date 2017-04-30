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
#define PyInt_AsLong   PyLong_AsLong
#define PyInt_Check    PyLong_Check

#define PyString_FromString        PyBytes_FromString
#define PyString_AsString          PyBytes_AsString
#define PyString_AsEncodedString   PyUnicode_AsEncodedString
#define PyString_Check             PyBytes_Check

#else // Python 2.7

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

#endif

#define  C_TEXT(text) ((char*)std::string(text).c_str())


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
#include "ROL_Algorithm.hpp"
#include "ROL_ElementwiseVector.hpp"
#include "ROL_EqualityConstraint.hpp"
#include "ROL_Objective.hpp"
#include "ROL_RandomVector.hpp"
#include "ROL_StdVector.hpp"


#endif // PYROL_HPP
