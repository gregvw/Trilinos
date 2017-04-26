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

#ifndef PYROL_EQUALITYCONSTRAINT_HPP
#define PYROL_EQUALITYCONSTRAINT_HPP

#include "PyROL.hpp"
#include "PyROL_AttributeManager.hpp"
#include "PyROL_BaseVector.hpp"

namespace PyROL {

/** \class PyROL::EqualityConstraint
    \brief Provides a ROL interface for equality constraint classes implemented in Python
*/

class EqualityConstraint : public ROL::EqualityConstraint<double>, public AttributeManager {

  using Vector = ROL::Vector<double>;

public:

  const static AttributeManager::Name className_;

private:

  const static AttributeManager::AttributeList attrList_;

  PyObject* pyEqCon_;


public:

  EqualityConstraint( PyObject* pyEqCon ) : 
    ROL::EqualityConstraint<double>(),
    AttributeManager( pyEqCon, attrList_, className_ ),
    pyEqCon_(pyEqCon) {
    Py_INCREF(pyEqCon_);
  }    

  virtual ~EqualityConstraint() {
    Py_DECREF(pyEqCon_);
  }
    
  virtual void value(Vector &c, const Vector &x, double &tol) {
    PyObject* pyC = Teuchos::dyn_cast<BaseVector>(c).getPyVector();
    const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
    PyObject* pyTol = PyFloat_FromDouble(tol);
    PyObject_CallMethodObjArgs(pyEqCon_,method_["value"].name,pyC,pyX,pyTol,NULL);
  }

  virtual void applyJacobian(Vector &jv, const Vector &v, const Vector &x, double &tol) {
    if( method_["applyJacobian"].impl ) {
      PyObject* pyJv = Teuchos::dyn_cast<BaseVector>(jv).getPyVector();
      const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();
      const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
      PyObject* pyTol = PyFloat_FromDouble(tol);
      PyObject_CallMethodObjArgs(pyEqCon_,method_["applyJacobian"].name,pyJv,pyV,pyX,pyTol,NULL);
      Py_DECREF(pyTol);
    }
    else {
      ROL::EqualityConstraint<double>::applyJacobian(jv, v, x, tol); 
    }
  }

  virtual void applyAdjointJacobian(Vector &ajv, const Vector &v, const Vector &x, double &tol) {
    if( method_["applyAdjointJacobian"].impl ) {
      PyObject* pyAjv = Teuchos::dyn_cast<BaseVector>(ajv).getPyVector();
      const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();
      const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
      PyObject* pyTol = PyFloat_FromDouble(tol);
      PyObject_CallMethodObjArgs(pyEqCon_,method_["applyAdjointJacobian"].name,pyAjv,pyV,pyX,pyTol,NULL);
      Py_DECREF(pyTol);
    }
    else {
      ROL::EqualityConstraint<double>::applyAdjointJacobian(ajv, v, x, tol); 
    }

  }

  // TODO: Add dual versions
  
  virtual void applyAdjointHessian(Vector &ahuv, const Vector &u, const Vector &v, const Vector &x, double &tol) {
    if( method_["applyAdjointHessian"].impl ) {
      PyObject* pyAhuv = Teuchos::dyn_cast<BaseVector>(ahuv).getPyVector();
      const PyObject* pyU = Teuchos::dyn_cast<const BaseVector>(u).getPyVector();
      const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();
      const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
      PyObject* pyTol = PyFloat_FromDouble(tol);
      PyObject_CallMethodObjArgs(pyEqCon_,method_["applyAdjointHessian"].name,pyAhuv,pyU,pyV,pyX,pyTol,NULL);
      Py_DECREF(pyTol);
    }
    else {
      ROL::EqualityConstraint<double>::applyAdjointHessian(ahuv, u, v, x, tol); 
    }
  }

  virtual std::vector<double> solveAugmentedSystem(Vector &v1,  Vector &v2, const Vector &b1,
                                                   const Vector &b2, const Vector &x, double &tol) {
    if( method_["solveAugmentedSystem"].impl ) {
      PyObject* pyV1 = Teuchos::dyn_cast<BaseVector>(v1).getPyVector();
      PyObject* pyV2 = Teuchos::dyn_cast<BaseVector>(v2).getPyVector();
      const PyObject* pyB1 = Teuchos::dyn_cast<const BaseVector>(b1).getPyVector();
      const PyObject* pyB2 = Teuchos::dyn_cast<const BaseVector>(b2).getPyVector();
      const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
      PyObject* pyTol = PyFloat_FromDouble(tol);
      PyObject_CallMethodObjArgs(pyEqCon_,method_["solveAugmentedSystem"].name,pyV1,pyV2,pyB1,pyB2,pyX,pyTol,NULL);
      Py_DECREF(pyTol);
      // TODO
      return std::vector<double>();
    }
    else {
      return ROL::EqualityConstraint<double>::solveAugmentedSystem(v1, v2, b1, b2, x, tol); 
    }
  }


  virtual void applyPreconditioner(Vector &pv, const Vector &v, const Vector &x, const Vector &g, double &tol) {
    if( method_["applyPreconditioner"].impl ) {
      PyObject* pyPv = Teuchos::dyn_cast<BaseVector>(pv).getPyVector();
      const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();
      const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
      const PyObject* pyG = Teuchos::dyn_cast<const BaseVector>(g).getPyVector();
      PyObject* pyTol = PyFloat_FromDouble(tol);
      PyObject_CallMethodObjArgs(pyEqCon_,method_["applyPreconditioner"].name,pyPv,pyV,pyX,pyG,pyTol,NULL);
      Py_DECREF(pyTol);
    }
    else {
      ROL::EqualityConstraint<double>::applyPreconditioner(pv, v, x, g, tol); 
    }
  }

  virtual void update( const Vector &x, bool flag = true, int iter = -1 ) {
    if( method_["update"].impl ) {
      const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
      PyObject* pyFlag = flag ? Py_True : Py_False;
      PyObject* pyIter = PyLong_FromLong(static_cast<long>(iter));
      PyObject_CallMethodObjArgs(pyEqCon_,method_["update"].name,pyX,pyFlag,pyIter,NULL);
      Py_DECREF(pyFlag);
      Py_DECREF(pyIter);
    }
  }

  virtual bool isFeasible( const Vector &v ) { 
    if( method_["isFeasible"].impl ) {
      const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();
      PyObject* pyFeasible = PyObject_CallMethodObjArgs(pyEqCon_,method_["isFeasible"].name,pyV,NULL);
      bool feasible = static_cast<bool>(PyLong_AsLong(pyFeasible));
      Py_DECREF(pyFeasible);
      return feasible;
    }
    else {
      return true; 
    }
  }



}; // class EqualityConstraint


} // namespace PyROL



#endif // PYROL_EQUALITYCONSTRAINT_HPP
