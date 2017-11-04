// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef PYROL_HPP
#define PYROL_HPP

#ifdef ENABLE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Python Includes
#include "Python.h"
#include "structmember.h"
#ifdef ENABLE_NUMPY
#include "numpy/arrayobject.h"
#endif


#if PY_MAJOR_VERSION >= 3

#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsLong   PyLong_AsLong
#define PyInt_Check    PyLong_Check

#define PyString_FromString        PyUnicode_FromString
// #define PyString_AsString          PyBytes_AsString
#define PyString_AsEncodedString   PyUnicode_AsEncodedString
#define PyString_Check             PyUnicode_Check

#include<string>
inline std::string PyString_AsString(PyObject* p)
{
  PyObject* pyString = PyUnicode_AsEncodedString(p,"ASCII","strict");
  std::string s = PyBytes_AsString(pyString);
  Py_XDECREF(pyString);
  return s;
}


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
#include "ROL_OptimizationSolver.hpp"

// Probably all of these are redundant 
#include "ROL_Algorithm.hpp"
#include "ROL_ElementwiseVector.hpp"
#include "ROL_EqualityConstraint.hpp"
#include "ROL_Objective.hpp"
#include "ROL_RandomVector.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_ValidParameters.hpp"


#endif // PYROL_HPP
