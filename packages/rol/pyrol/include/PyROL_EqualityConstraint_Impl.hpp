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

#ifndef PYROL_EQUALITYCONSTRAINT_IMPL_HPP
#define PYROL_EQUALITYCONSTRAINT_IMPL_HPP

#include "PyROL_EqualityConstraint.hpp"
#include "PyROL_TypeConverters.hpp"

namespace PyROL {

/** \class PyROL::EqualityConstraint
    \brief Provides a ROL interface for equality constraint classes implemented in Python
*/


EqualityConstraint::EqualityConstraint( PyObject* pyEqCon ) : 
    ROL::EqualityConstraint<double>(),
  AttributeManager( pyEqCon, attrList_, className_ ),
  pyEqCon_(pyEqCon) {
  Py_INCREF(pyEqCon_);
}    

EqualityConstraint::~EqualityConstraint() {
  Py_DECREF(pyEqCon_);
}

void EqualityConstraint::value(ROL::Vector<double> &c, const ROL::Vector<double> &x, double &tol) {
  PyObject* pyC = PyObject_FromVector(c);
  const PyObject* pyX = PyObject_FromVector(x);
  PyObject* pyTol = PyFloat_FromDouble(tol);
  PyObject_CallMethodObjArgs(pyEqCon_,method_["value"].name,
    pyC,pyX,pyTol,NULL);
}

void EqualityConstraint::applyJacobian(ROL::Vector<double> &jv, const ROL::Vector<double> &v, 
                                       const ROL::Vector<double> &x, double &tol) {
  if( method_["applyJacobian"].impl ) {
    PyObject* pyJv = PyObject_FromVector(jv);
    const PyObject* pyV = PyObject_FromVector(v);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyEqCon_,method_["applyJacobian"].name,
      pyJv,pyV,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    ROL::EqualityConstraint<double>::applyJacobian(jv, v, x, tol); 
  }
}

void EqualityConstraint::applyAdjointJacobian(ROL::Vector<double> &ajv, const ROL::Vector<double> &v, 
                                              const ROL::Vector<double> &x, double &tol) {
  if( method_["applyAdjointJacobian"].impl ) {
    PyObject* pyAjv = PyObject_FromVector(ajv);
    const PyObject* pyV = PyObject_FromVector(v);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyEqCon_,method_["applyAdjointJacobian"].name,
      pyAjv,pyV,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    ROL::EqualityConstraint<double>::applyAdjointJacobian(ajv, v, x, tol); 
  }

}

// TODO: Add dual versions

void EqualityConstraint::applyAdjointHessian(ROL::Vector<double> &ahuv, const ROL::Vector<double> &u, 
                                             const ROL::Vector<double> &v, const ROL::Vector<double> &x, double &tol) {
  if( method_["applyAdjointHessian"].impl ) {
    PyObject* pyAhuv = PyObject_FromVector(ahuv);
    const PyObject* pyU = PyObject_FromVector(u);
    const PyObject* pyV = PyObject_FromVector(v);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyEqCon_,method_["applyAdjointHessian"].name,
      pyAhuv,pyU,pyV,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    ROL::EqualityConstraint<double>::applyAdjointHessian(ahuv, u, v, x, tol); 
  }
}

std::vector<double> 
EqualityConstraint::solveAugmentedSystem(ROL::Vector<double> &v1,  ROL::Vector<double> &v2, 
                                         const ROL::Vector<double> &b1, const ROL::Vector<double> &b2, 
                                         const ROL::Vector<double> &x, double &tol) {
  if( method_["solveAugmentedSystem"].impl ) {
    PyObject* pyV1 = PyObject_FromVector(v1);
    PyObject* pyV2 = PyObject_FromVector(v2);
    const PyObject* pyB1 = PyObject_FromVector(b1);
    const PyObject* pyB2 = PyObject_FromVector(b2);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyEqCon_,method_["solveAugmentedSystem"].name,
      pyV1,pyV2,pyB1,pyB2,pyX,pyTol,NULL);
    Py_DECREF(pyTol);
    // TODO
    return std::vector<double>();
  }
  else {
    return ROL::EqualityConstraint<double>::solveAugmentedSystem(v1, v2, b1, b2, x, tol); 
  }
}


void EqualityConstraint::applyPreconditioner(ROL::Vector<double> &pv, const ROL::Vector<double> &v, 
                                             const ROL::Vector<double> &x, const ROL::Vector<double> &g, 
                                             double &tol) {
  if( method_["applyPreconditioner"].impl ) {
    PyObject* pyPv = Teuchos::dyn_cast<BaseVector>(pv).getPyVector();
    const PyObject* pyV = PyObject_FromVector(v);
    const PyObject* pyX = PyObject_FromVector(x);
    const PyObject* pyG = PyObject_FromVector(g);
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyEqCon_,method_["applyPreconditioner"].name,
      pyPv,pyV,pyX,pyG,pyTol,NULL);
    Py_DECREF(pyTol);
  }
  else {
    ROL::EqualityConstraint<double>::applyPreconditioner(pv, v, x, g, tol); 
  }
}

void EqualityConstraint::update( const ROL::Vector<double> &x, 
                                 bool flag, int iter ) {
  if( method_["update"].impl ) {
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyFlag = flag ? Py_True : Py_False;
    PyObject* pyIter = PyLong_FromLong(static_cast<long>(iter));
    PyObject_CallMethodObjArgs(pyEqCon_,method_["update"].name,
      pyX,pyFlag,pyIter,NULL);
    Py_DECREF(pyFlag);
    Py_DECREF(pyIter);
  }
}

bool EqualityConstraint::isFeasible( const ROL::Vector<double> &v ) { 
  if( method_["isFeasible"].impl ) {
    const PyObject* pyV = PyObject_FromVector(v);
    PyObject* pyFeasible = PyObject_CallMethodObjArgs(pyEqCon_,
      method_["isFeasible"].name,pyV,NULL);
    bool feasible = static_cast<bool>(PyLong_AsLong(pyFeasible));
    Py_DECREF(pyFeasible);
    return feasible;
  }
  else {
    return true; 
  }
}

} // namespace PyROL



#endif // PYROL_EQUALITYCONSTRAINT_IMPL_HPP
