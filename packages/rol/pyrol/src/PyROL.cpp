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


#include "test/PyROL_TestVector.hpp"
#include "test/PyROL_TestObjective.hpp"
#include "test/PyROL_TestEqualityConstraint.hpp"
#include "PyROL_SolveUnconstrained.hpp"
#include "PyROL_SolveEqualityConstrained.hpp"

#ifdef __cplusplus
extern "C" {
#endif


static PyMethodDef pyrol_methods[] = {
  {"testVector",(PyCFunction)testVector,METH_VARARGS,testVector_doc},
  {"testObjective",(PyCFunction)testObjective,METH_VARARGS,testObjective_doc},
  {"testEqualityConstraint",(PyCFunction)testEqualityConstraint,METH_VARARGS,
    testEqualityConstraint_doc},
  {"solveUnconstrained",(PyCFunction)solveUnconstrained,METH_VARARGS,
    solveUnconstrained_doc},
  {"solveEqualityConstrained",(PyCFunction)solveEqualityConstrained,
    METH_VARARGS,solveEqualityConstrained_doc},
  {NULL, NULL, 0, NULL}
};

static char pyrol_doc[] =
  "PyROL: the Python interface to the Rapid Optimization Library";

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pyrol_module = {
  PyModuleDef_HEAD_INIT,
  "pyrol",
  pyrol_doc,
  -1,
  pyrol_methods
};
#endif



#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_pyrol(void) {
#ifdef ENABLE_NUMPY
  import_array();
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module pyrol (failed to import numpy)");

#endif
  PyObject* mod = PyModule_Create(&pyrol_module);
  return mod;
}

#else
void initpyrol(void) {
#ifdef ENABLE_NUMPY
  import_array();
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module pyrol (failed to import numpy)");

#endif
  Py_InitModule3("pyrol",pyrol_methods,pyrol_doc);
}
#endif


#ifdef __cplusplus
} // extern "C"
#endif
