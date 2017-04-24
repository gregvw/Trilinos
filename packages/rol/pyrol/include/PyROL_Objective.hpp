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

#ifndef PYROL_OBJECTIVE_HPP
#define PYROL_OBJECTIVE_HPP

#include "PyROL.hpp"
#include "PyROL_AttributeManager.hpp"
#include "PyROL_BaseVector.hpp"

namespace PyROL {

/** \class PyROL::Objective
    \brief Provides a ROL interface for objective classes implemented in Python
           Note that the Python-implemented objective has to be able to specify
           its concrete vector type
*/

class Objective : public ROL::Objective<double>, public AttributeManager {

  using Vector = ROL::Vector<double>;

public:

  const static AttributeManager::Name className_;

private:

  const static AttributeManager::AttributeList attrList_;

  PyObject* pyObjective_;

public:

 Objective( PyObject* pyObjective ) :
   AttributeManager( pyObjective, attrList_, className_ ),
   pyObjective_(pyObjective) {
   Py_INCREF(pyObjective_);
 }

 virtual ~Objective() {
   Py_DECREF(pyObjective_);
 }

 virtual void update( const Vector &x, bool flag = true, int iter = -1 ) {
   if( method_["update"].impl ) {
     const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
     PyObject* pyFlag = flag ? Py_True : Py_False;
     PyObject* pyIter = PyLong_FromLong(static_cast<long>(iter));
     
     PyObject_CallMethodObjArgs(pyObjective_,method_["update"].name,pyX,pyFlag,pyIter,NULL);
     
     Py_DECREF(pyFlag);
     Py_DECREF(pyIter);
   }
 }

 virtual double value( const Vector &x, double &tol ) {
   const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
   PyObject* pyTol = PyFloat_FromDouble(tol);
   PyObject* pyValue = PyObject_CallMethodObjArgs(pyObjective_,method_["value"].name,pyX,pyTol,NULL);
   double val = PyFloat_AsDouble(pyValue);
   Py_DECREF(pyTol);
   Py_DECREF(pyValue);
   return val;
 }

 virtual void gradient( Vector &g, const Vector &x, double &tol ) {
   if( method_["gradient"].impl ) {
     PyObject* pyG = Teuchos::dyn_cast<BaseVector>(g).getPyVector();
     const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
     PyObject* pyTol = PyFloat_FromDouble(tol);
     PyObject_CallMethodObjArgs(pyObjective_,method_["gradient"].name,pyG,pyX,pyTol,NULL);
   }
   else {
     ROL::Objective<double>::gradient(g,x,tol);
   } 
 }

 virtual double dirDeriv( const Vector &x, const Vector &d, double &tol ) {
   if( method_["dirDeriv"].impl ) {
     const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
     const PyObject* pyD = Teuchos::dyn_cast<const BaseVector>(d).getPyVector();
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
  
 virtual void hessVec( Vector &hv, const Vector &v, const Vector &x, double &tol ) {
   if( method_["hessVec"].impl ) {
     PyObject* pyHv = Teuchos::dyn_cast<BaseVector>(hv).getPyVector();
     const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();     
     const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
     PyObject* pyTol = PyFloat_FromDouble(tol);
     PyObject_CallMethodObjArgs(pyObjective_,method_["hessVec"].name,pyHv,pyV,pyX,pyTol,NULL);
     Py_DECREF(pyTol);
   }
   else {
     return ROL::Objective<double>::hessVec(hv,v,x,tol);
   }
 }

 virtual void invHessVec( Vector &hv, const Vector &v, const Vector &x, double &tol ) {
   if( method_["invHessVec"].impl ) {
     PyObject* pyHv = Teuchos::dyn_cast<BaseVector>(hv).getPyVector();
     const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();     
     const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
     PyObject* pyTol = PyFloat_FromDouble(tol);
     PyObject_CallMethodObjArgs(pyObjective_,method_["invHessVec"].name,pyHv,pyV,pyX,pyTol,NULL);
     Py_DECREF(pyTol);
   }
   else {
     return ROL::Objective<double>::invHessVec(hv,v,x,tol);
   }
 }

 virtual void precond( Vector &Pv, const Vector &v, const Vector &x, double &tol ) {
   if( method_["precond"].impl ) {
     PyObject* pyPv = Teuchos::dyn_cast<BaseVector>(Pv).getPyVector();
     const PyObject* pyV = Teuchos::dyn_cast<const BaseVector>(v).getPyVector();     
     const PyObject* pyX = Teuchos::dyn_cast<const BaseVector>(x).getPyVector();
     PyObject* pyTol = PyFloat_FromDouble(tol);
     PyObject_CallMethodObjArgs(pyObjective_,method_["precond"].name,pyPv,pyV,pyX,pyTol,NULL);
     Py_DECREF(pyTol);
   }
   else {
     return ROL::Objective<double>::precond(Pv,v,x,tol);
   }
 }

}; // class Objective


} // namespace PyROL


#endif // PYROL_OBJECTIVE_HPP
