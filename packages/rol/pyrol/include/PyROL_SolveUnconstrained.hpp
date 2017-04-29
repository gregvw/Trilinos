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

#ifndef PYROL_SOLVE_UNCONSTRAINED_HPP
#define PYROL_SOLVE_UNCONSTRAINED_HPP

#include "PyROL.hpp"
#include "PyROL_Objective.hpp"

/* Solve an unconstrained problem, given an objective, initial guess, and
 * options */

static PyObject* solveUnconstrained( PyObject* self, PyObject* pyArgs ) {

  PyObject* pyObjective;
  PyObject* pyX;
  PyObject* pyOptions;

  if( !PyArg_ParseTuple(pyArgs,"OOO",&pyObjective,&pyX,&pyOptions) ) return NULL;

  // Make a PyROL::Objective from the supplied Python objective
  PyROL::Objective obj(pyObjective);

  // Get optimization vector
  Teuchos::RCP<ROL::Vector<double> > xp = PyROL::PyObject_AsVector(pyX) ;  

  // Get parameters
  Teuchos::ParameterList parlist;
  PyROL::dictToParameterList(pyOptions,parlist);

  std::cout << parlist << std::endl;

  std::string algoKey("Algorithm");
  PyObject *pyAlgoKey = PyString_FromString(C_TEXT(algoKey));

  // Borrowed reference
  PyObject *pyAlgoValue = PyDict_GetItem(pyOptions,pyAlgoKey);
  PyObject *pyTemp = PyString_AsEncodedString(pyAlgoValue,"ASCII","strict");
  Py_DECREF(pyAlgoKey); 

  std::string algoValue = PyString_AsString(pyTemp);
 
  // Capture stream from Algorithm::run in string
  std::stringstream ss;

  // Build and run the algorithm
  ROL::Algorithm<double> algo(algoValue,parlist,false);
  algo.run(*xp,obj,true,ss);

  PyObject* pyOutput = PyString_FromString(C_TEXT(ss.str()));

  return pyOutput;

} // solveUnconstrained

static char solveUnconstrained_doc[] = 
  "solveUnconstrained(obj,x,options) : Solve an unconstrained optimization "
  "problem using PyROL, where obj is an Objective function class, x is the "
  "initial guess, and options is a dictionary of options for configuring "
  "ROL's minimization algorithm.";



#endif // PYROL_SOLVE_UNCONSTRAINED_HPP

