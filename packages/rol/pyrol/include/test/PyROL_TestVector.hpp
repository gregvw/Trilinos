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

#ifndef PYROL_TESTVECTOR_HPP
#define PYROL_TESTVECTOR_HPP

#include "PyROL_PythonVector.hpp"

// TODO: This needs to be generalized for alterative concrete vector types
static PyObject* testVector( PyObject* self, PyObject *pyArgs ) {

  PyObject* pyVector;

  if( !PyArg_ParseTuple(pyArgs,"O",&pyVector) ) return NULL;

  PyROL::PythonVector x(pyVector);

  Teuchos::RCP<ROL::Vector<double>> y = x.clone();
  Teuchos::RCP<ROL::Vector<double>> z = y->clone();
  
  ROL::RandomizeVector(*y);
  ROL::RandomizeVector(*z);
 
  std::stringstream ss;

  std::vector<double> vcheck;

  vcheck = x.checkVector(*y,*z,true,ss);  

  PyObject *pyList = PyList_New(vcheck.size());  

  std::string output = ss.str();

  PyObject* pyOutput = PyString_FromString(C_TEXT(output));

  PyObject* pyReturn = PyTuple_New(2);
  
  Py_ssize_t index = 0;
 
  for( auto element : vcheck ) {
    PyList_SetItem(pyList,index,PyFloat_FromDouble(element));
    index++;
  }
  PyTuple_SetItem(pyReturn,(Py_ssize_t)(0),pyList);
  PyTuple_SetItem(pyReturn,(Py_ssize_t)(1),pyOutput);
  return pyReturn;
}

static char testVector_doc[] = 
  "vcheck,output = testVector(x) : Perform standard vector arithmetic tests on the given "
  "vector x. \n" 
  "Returns the tuple with elements: \n"
  "   vcheck - list of values indicating the error from each test. \n"
  "   output - detailed information about the tests that were run "; 

#endif // PYROL_TESTVECTOR_HPP
