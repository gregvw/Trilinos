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

#ifndef PYROL_TESTEQUALITYCONSTRAINT_HPP
#define PYROL_TESTEQUALITYCONSTRAINT_HPP

#include "PyROL_TypeConverters.hpp"
#include "PyROL_EqualityConstraint.hpp"

static PyObject* testEqualityConstraint( PyObject* self, PyObject* pyArgs ) {

  PyObject* pyEqualityConstraint;
  PyObject* pyX;                       // Optimization vector
  PyObject* pyL;                       // Lagrange multiplier
  PyObject* pyOptions;

  if( !PyArg_ParseTuple(pyArgs,"OOOO",&pyEqualityConstraint, &pyX, &pyL, &pyOptions) ) return NULL;

  PyROL::EqualityConstraint eqcon(pyEqualityConstraint);

  auto x = PyROL::PyObject_AsVector(pyX);
  auto l = PyROL::PyObject_AsVector(pyL);

  auto d  = x->clone();
  auto v  = x->clone();
  auto jv = l->dual().clone();
  auto w  = l->dual().clone();
  
  ROL::RandomizeVector(*d);
  ROL::RandomizeVector(*v);
  ROL::RandomizeVector(*w);  

  int order = 1;
  int numSteps = 10;
  std::vector<double> steps;

  // Get options
  PyObject* pyValue; 

  pyValue = PyDict_GetItemString(pyOptions,"Order");

  if( pyValue != NULL ) {
    order = static_cast<int>(PyLong_AsLong(pyValue));
  }

  pyValue = PyDict_GetItemString(pyOptions,"Steps");
  if( pyValue != NULL ) {

    Py_ssize_t pyLength = PyList_Size(pyValue);

    numSteps = static_cast<int>(pyLength);
    int i=0;

    for(Py_ssize_t pyIndex = 0; pyIndex<pyLength; ++pyIndex) {
      PyObject* pyElement = PyList_GetItem(pyValue, pyIndex);
      double element = PyFloat_AsDouble(pyElement);
      steps.push_back(element);
      ++i;
    }     
  }
  else {

    pyValue = PyDict_GetItemString(pyOptions,"Number of Steps");

    if( pyValue != NULL ) {
      numSteps = static_cast<int>(PyLong_AsLong(pyValue));
    } 
    else {
      numSteps = ROL_NUM_CHECKDERIV_STEPS;
    }

    for(int i=0; i<numSteps; ++i) {
      steps.push_back(std::pow(10,-i));
    }
  }

  // Capture output of checks
  std::stringstream outputStream;

  // Finite difference check of constraint Jacobian
  pyValue = PyDict_GetItemString(pyOptions,"Check Jacobian");
  
  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {

    if( eqcon.hasImplementation("applyJacobian") ) {
      outputStream << "\n\nConstraint Jacobian Check\n";
      eqcon.checkApplyJacobian( *x, *v, *jv, steps, true, outputStream, order ); 
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot check "
        "applyJacobian. EqualityConstraint does not implement applyJacobian");
    }
  } 

  // Check adjoint consistency
  pyValue = PyDict_GetItemString(pyOptions,"Check Adjoint Jacobian Consistency");
  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {
    if( eqcon.hasImplementation("applyJacobian") && 
        eqcon.hasImplementation("applyAdjointJacobian") ) {
        eqcon.checkAdjointConsistencyJacobian(*w,*v,*x,true,outputStream);
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot check "
        "adjoint Jacobian consistency. EqualityConstraint does not implement "
        "both applyJacobian and applyAdjointJacobian");
    }
  }

  // Check adjoint Hessian
  pyValue = PyDict_GetItemString(pyOptions,"Check Adjoint Hessian");
  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {
    if(eqcon.hasImplementation("applyAdjointHessian")) {
      auto u = l->dual().clone();
      auto ahuv = x->dual().clone();
 
      ROL::RandomizeVector(*u);
      ROL::RandomizeVector(*ahuv);

      outputStream << "\n\nConstraint Adjoint Hessian Check\n";
      eqcon.checkApplyAdjointHessian(*x,*u,*v,*ahuv,true,outputStream); 
    }
    else {
       TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot check "
        "applyAdjointHessian. EqualityConstraint does not implement applyHessian");
    } 
  }

  PyObject* pyOutput = PyString_FromString(C_TEXT(outputStream.str()));

  return pyOutput; 

}

static char testEqualityConstraint_doc[] = 
  "testEqualityConstraint(eqcon,x,l,options) : Use finite difference (FD) methods to validate equality "
  "constraint methods involving derivatives if they have been implemented.\n\n "
  "Arguments:\n"
  "eqcon   - the equality constraint object \n"
  "x       - the optimization vector about which to evaluate the equality constraint and its methods\n"
  "l       - the Lagrange multiplier vector\n"
  "options - dictionary specifying how to do the checks. Valid keys and values below. \n\n"
  "    Key                                   | Value Type      | Function                               \n"
  "    --------------------------------------+-----------------+----------------------------------------\n"
  "   \"Check Jacobian\"                     | bool            | Checks directional derivative if       \n" 
  "                                          |                 | applyJacobian is implemented           \n"
  "   \"Check Adjoint Jacobian Consistency\" | bool            | Checks if <w,Jv> is sufficiently close \n"
  "                                          |                 | in angle to <adj(J)w,v>                \n"
  "   \"Check Adjoint Hessian\"              | bool            | Checks the v-directional derivative of \n"           "                                          |                 | the adjoint Jacobian evaluated at x    \n"
  "                                          |                 | applied to the direction vector u      \n"
  "   \"Order\"                              | int             | FD order (valid values are 1,2,3,4)    \n"
  "   \"Number of Steps\"                    | int             | Number of FD steps to sweep over       \n" 
  "   \"Steps\"                              | list of double  | User-defined step sizes to compute FD    ";



#endif // PYROL_TESTEQUALITYCONSTRAINT_HPP

