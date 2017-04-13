#ifndef PYROL_HPP
#define PYROL_HPP

// Python Includes
#include "Python.h"

#if PY_MAJOR_VERSION >= 3

#define PyInt_FromLong PyLong_FromLong
#define PyString_FromString PyUnicode_FromString

#endif

// ROL Includes
#include "ROL_Vector.hpp"

// PyROL Includes 
#include "PyROL_AttributeManager.hpp"

#endif // PYROL_HPP
