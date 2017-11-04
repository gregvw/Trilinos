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

#ifndef PYROL_TESTOBJECTIVE_HPP
#define PYROL_TESTOBJECTIVE_HPP

#include "PyROL_TypeConverters.hpp"
#include "PyROL_Objective.hpp"

static PyObject* testObjective( PyObject* self, PyObject *pyArgs ) {

  using RV = ROL::Vector<double>;

  PyObject* pyObjective;
  PyObject* pyX;
  PyObject* pyOptions;

  if( !PyArg_ParseTuple(pyArgs,"OOO",&pyObjective,&pyX,&pyOptions) ) return NULL; 

  // Make a PyROL::Objective from the supplied Python objective
  PyROL::Objective obj(pyObjective);

  // Get optimization vector
  Teuchos::RCP<RV> x = PyROL::PyObject_AsVector(pyX);  

  Teuchos::RCP<RV> d = x->clone();
  Teuchos::RCP<RV> v = x->clone();   

  ROL::RandomizeVector(*d,-1.0,1.0);
  ROL::RandomizeVector(*v,-1.0,1.0);
//  v->applyUnary(ROL::Elementwise::Fill<double>(1.0));

  // Finite difference order
  int order = 1;

  // Finite difference steps
  std::vector<double> steps;

  // Number of finite difference steps
  int numSteps = 10;

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

  // Gradient finite difference check 
  pyValue = PyDict_GetItemString(pyOptions,"Check Gradient");

  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {

    if( obj.hasImplementation("gradient") ) {
      outputStream << "\n\nObjective Gradient Check\n";
      obj.checkGradient(*x,*d,steps,true,outputStream,order);   
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot gradient check. "
        "Objective does not implement a gradient method.");
    }
  }

  // Hessian-Vector Product finite difference check 
  pyValue = PyDict_GetItemString(pyOptions,"Check HessVec");

  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {
  
    if( obj.hasImplementation("hessVec") ) {
      outputStream << "\n\nObjective Hessian-Vector Product Check\n";
      obj.checkHessVec(*x,*v,steps,true,outputStream,order); 
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot check Hessian-vector product. "
        "Objective does not implement a hessVec method.");
    }
  } 

  // Hessian symmetry check
  pyValue = PyDict_GetItemString(pyOptions,"Check HessSym");

  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {
  
    if( obj.hasImplementation("hessVec") ) {
      outputStream << "\n\nObjective Hessian Symmetry Check\n";
      obj.checkHessSym(*x,*d,*v,true,outputStream);
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot check Hessian-vector product. "
        "Objective does not implement a hessVec method.");
    }
  } 

  // Check inverse Hessian-vector product with Hessian
  pyValue = PyDict_GetItemString(pyOptions,"Check InvHessVec"); 

  if( pyValue != NULL && PyObject_IsTrue(pyValue) ) {
    if( obj.hasImplementation("invHessVec") ) {
      Teuchos::RCP<RV> hv   = x->clone();
      Teuchos::RCP<RV> ihhv = x->clone();
      double tol = std::sqrt(ROL::ROL_EPSILON<double>());
      obj.hessVec(*hv, *v, *x, tol);
      obj.invHessVec(*ihhv, *hv, *x, tol);

      ihhv->axpy(-1.0,*v);
      outputStream << std::endl;
      outputStream << "Checking Objective inverse Hessian" << std::endl;
      outputStream << "||H^{-1}Hv-v|| = " << ihhv->norm() << std::endl;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error, "Cannot check inverse Hessian-vector product. "
        "Objective does not implement a invHessVec method.");
    }
  }

  PyObject* pyOutput  = PyString_FromString(C_TEXT(outputStream.str())); 

  return pyOutput;
}

static char testObjective_doc[] = 
  "testObjective(obj,x,options) : Use finite difference (FD) methods to validate objective methods "
  "involving derivatives if they have been implemented.\n\n "
  "Arguments:\n"
  "obj     - the objective function \n"
  "x       - the optimization vector about which to evaluate the objective and its methods\n"
  "options - dictionary specifying how to do the checks. Valid keys and values below. \n\n"
  "    Key                 | Value Type      | Function \n"
  "    --------------------+-----------------+-------------------------------------------------------\n"
  "   \"Check Gradient\"   |  bool           | Checks directional derivative if gradient implemented\n"
  "   \"Check HessVec\"    |  bool           | Checks Hessian-Vector product if hessVec implemented\n"
  "   \"Check HessSym\"    |  bool           | Checks symmetry of the Hessian if hessVec implemented\n"
  "   \"Check InvHessVec\" |  bool           | Checks if H^(-1)Hv = v if invHessVec and hessVec exist\n"
  "   \"Order\"            |  int            | FD order (valid values are 1,2,3,4)\n"
  "   \"Number of Steps\"  |  int            | Number of FD steps to sweep over\n" 
  "   \"Steps\"            |  list of double | User-defined step sizes to compute FD approximations\n\n";




#endif // PYROL_TESTOBJECTIVE_HPP

