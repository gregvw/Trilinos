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

#ifndef PYROL_OBJECTIVE_IMPL_HPP
#define PYROL_OBJECTIVE_IMPL_HPP

#include "PyROL_Objective.hpp"
#include "PyROL_TypeConverters.hpp"

namespace PyROL {

/** \class PyROL::Objective
    \brief Provides a ROL interface for objective classes implemented in Python
*/

Objective::Objective( PyObject* pyObjective ) :
  AttributeManager( pyObjective, attrList_, className_ ),
  pyObjective_(pyObjective) {
  Py_INCREF(pyObjective_);
}

Objective::~Objective() {
  Py_DECREF(pyObjective_);
}

void Objective::update( const ROL::Vector<double> &x, bool flag, int iter ) {
  if( method_["update"].impl ) {
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyFlag = flag ? Py_True : Py_False;
    PyObject* pyIter = PyLong_FromLong(static_cast<long>(iter));

    PyObject_CallMethodObjArgs(pyObjective_,method_["update"].name,pyX,pyFlag,pyIter,NULL);

    Py_DECREF(pyFlag);
    Py_DECREF(pyIter);
  }
}

double Objective::value( const ROL::Vector<double> &x, double &tol ) {
  const PyObject* pyX = PyObject_FromVector(x);
  PyObject* pyTol = PyFloat_FromDouble(tol);
  PyObject* pyValue = PyObject_CallMethodObjArgs(pyObjective_,method_["value"].name,pyX,pyTol,NULL);
  TEUCHOS_TEST_FOR_EXCEPTION(!PyFloat_Check(pyValue), std::logic_error,
                             "value() returned incorrect type");
  double val = PyFloat_AsDouble(pyValue);
  Py_DECREF(pyTol);
  Py_DECREF(pyValue);
  return val;
}

void Objective::gradient( ROL::Vector<double> &g, const ROL::Vector<double> &x, double &tol ) {
  if( method_["gradient"].impl ) {
    PyObject* pyG = PyObject_FromVector(g);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyObjective_,method_["gradient"].name,pyG,pyX,pyTol,NULL);
  }
  else {
    ROL::Objective<double>::gradient(g,x,tol);
  }
}

double Objective::dirDeriv( const ROL::Vector<double> &x, const ROL::Vector<double> &d, double &tol ) {
  if( method_["dirDeriv"].impl ) {
    const PyObject* pyX = PyObject_FromVector(x);
    const PyObject* pyD = PyObject_FromVector(d);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject* pyResult = PyObject_CallMethodObjArgs(pyObjective_,method_["dirDeriv"].name,pyX,pyD,pyTol,NULL);
    double result = PyFloat_AsDouble(pyResult);
    Py_DECREF(pyTol);
    Py_DECREF(pyResult);
    return result;
  }
  else {
    return ROL::Objective<double>::dirDeriv(x,d,tol);
  }
}

void Objective::hessVec( ROL::Vector<double> &hv, const ROL::Vector<double> &v, const ROL::Vector<double> &x, double &tol ) {
  if( method_["hessVec"].impl ) {
    PyObject* pyHv = Teuchos::dyn_cast<BaseVector>(hv).getPyVector();
    const PyObject* pyV = PyObject_FromVector(v);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyObjective_,method_["hessVec"].name,pyHv,pyV,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    return ROL::Objective<double>::hessVec(hv,v,x,tol);
  }
}

void Objective::invHessVec( ROL::Vector<double> &hv, const ROL::Vector<double> &v, const ROL::Vector<double> &x, double &tol ) {
  if( method_["invHessVec"].impl ) {
    PyObject* pyHv = Teuchos::dyn_cast<BaseVector>(hv).getPyVector();
    const PyObject* pyV = PyObject_FromVector(hv);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyObjective_,method_["invHessVec"].name,pyHv,pyV,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    return ROL::Objective<double>::invHessVec(hv,v,x,tol);
  }
}

void Objective::precond( ROL::Vector<double> &Pv, const ROL::Vector<double> &v, const ROL::Vector<double> &x, double &tol ) {
  if( method_["precond"].impl ) {
    PyObject* pyPv = PyObject_FromVector(Pv);
    const PyObject* pyV = PyObject_FromVector(v);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyObjective_,method_["precond"].name,pyPv,pyV,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    return ROL::Objective<double>::precond(Pv,v,x,tol);
  }
}


} // namespace PyROL


#endif // PYROL_OBJECTIVE_HPP
