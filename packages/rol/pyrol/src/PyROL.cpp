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


#include "PyROL_TestVector.hpp"

#include <iostream>

// Basic placeholder code to verify that CMake and Python are 
// playing nice 
extern "C" {

static PyObject * 
display( PyObject *self, PyObject *args ) {
  char *myString;
   if( !PyArg_ParseTuple(args,"s",&myString) )
     return NULL;
   std::cout << myString << std::endl;
   Py_INCREF(Py_None);
   return Py_None;
}

static char display_doc[] = 
  "display( ): Output supplied string to console.\n";

static PyMethodDef pyrol_methods[] = {
  {"display", (PyCFunction)display,METH_VARARGS,display_doc},
  {"testVector",(PyCFunction)testVector,METH_VARARGS,testVector_doc},
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


PyMODINIT_FUNC 
#if PY_MAJOR_VERSION >= 3
PyInit_pyrol(void) {
  return PyModule_Create(&pyrol_module);
}
#else
initpyrol(void) {
  Py_InitModule3("pyrol",pyrol_methods,pyrol_doc);
}
#endif


} // extern "C"
