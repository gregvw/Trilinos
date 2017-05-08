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

#ifndef PYROL_SOLVE_EQUALITYCONSTRAINED_HPP
#define PYROL_SOLVE_EQUALITYCONSTRAINED_HPP

#include "PyROL.hpp"
#include "PyROL_Objective.hpp"
#include "PyROL_EqualityConstraint.hpp"

/* Solve an equality-constrained optimization problem, given an objective,
   equality constraint, initial guess, and options
 */
static PyObject* solveEqualityConstrained( PyObject* self, PyObject* pyArgs ) {

  using namespace PyROL;

  PyObject* pyObjective;
  PyObject* pyEqualityConstraint;
  PyObject* pyX;
  PyObject* pyL;
  PyObject* pyOptions;
  PyObject* pyReturn;
  pyReturn = PyTuple_New(2);

  int parseCheck = PyArg_ParseTuple(pyArgs,"OOOOO",&pyObjective,&pyEqualityConstraint,
                   &pyX,&pyL,&pyOptions);
  TEUCHOS_TEST_FOR_EXCEPTION(!parseCheck,std::logic_error,"Failed to parse input tuple."
    << " Expected solveEqualityConstrained(obj,con,x,l,opt)");

  // Make a PyROL::Objective from the supplied Python object
  PyROL::Objective obj(pyObjective);

  // Make a PyROL::EqualityConstraint from the supplied Python object
  PyROL::EqualityConstraint con(pyEqualityConstraint);

  // Get optimization vector
  Teuchos::RCP<ROL::Vector<double> > xp = PyObject_AsVector(pyX) ;

  // Get multiplier vector
  Teuchos::RCP<ROL::Vector<double> > lp = PyObject_AsVector(pyL) ;

  // Get parameters
  Teuchos::ParameterList parlist;
  PyROL::dictToParameterList(pyOptions,parlist);

  std::string algoKey("Algorithm");
  PyObject *pyAlgoKey = PyUnicode_FromString((char*)algoKey.c_str());

  // Borrowed reference
  PyObject *pyAlgoValue = PyDict_GetItem(pyOptions,pyAlgoKey);
  Py_DECREF(pyAlgoKey);
  //  PyObject* pyTemp = PyString_AsEncodedString(pyAlgoValue,"ASCII","strict");

  std::string algoValue = PyString_AsString(pyAlgoValue);

  // Capture streams from Algorithm::run in strings
  std::stringstream outputStream;
  std::stringstream vectorStream;

  ROL::Algorithm<double> algo(algoValue,parlist,false);

  algo.run(*xp,*lp,obj,con,true);

  PyObject* pyOutput  = PyString_FromString(C_TEXT(outputStream.str()));
  PyObject* pyVectors = PyString_FromString(C_TEXT(vectorStream.str()));

  PyTuple_SetItem(pyReturn,(Py_ssize_t)(0),pyOutput);
  PyTuple_SetItem(pyReturn,(Py_ssize_t)(1),pyVectors);

  return pyReturn;
}


static char solveEqualityConstrained_doc[] =
  "solveE(obj,con,x,l,options) : Solve an  equality constrained "
  "optimization problem using PyROL, where obj is an Objective "
  "function class, con is an EqualityConststraint class, x is the "
  "initial guess of the optimization vector, l vector of Lagrange "
  "multipliers and options is a dictionary of options for configuring "
  "ROL's minimization algorithm.";


#endif // PYROL_SOLVE_EQUALITYCONSTRAINED_HPP
