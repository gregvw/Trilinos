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


#ifndef PYROL_PYTHONVECTOR_HPP
#define PYROL_PYTHONVECTOR_HPP

#include "PyROL_AttributeManager.hpp"

/** \class PyROL::PythonVector
 *  \brief Provides a ROL interface to generic vectors defined in Python 
 *         which satisfy the interface requirements
 */

namespace PyROL {

class PythonVector : public ROL::ElementwiseVector<double>, public AttributeManager {

  using Vector          = ROL::Vector<double>;

  using UnaryFunction   = ROL::Elementwise::UnaryFunction<double>;  
  using BinaryFunction  = ROL::Elementwise::BinaryFunction<double>;  
  using ReductionOp     = ROL::Elementwise::ReductionOp<double>;  

private: 
  
  const static AttributeManager::AttributeList attrList_;

  PyObject* pyVector_;

  bool has_ownership_;

public:

  PythonVector( PyObject* pyVector, bool has_ownership=false ) : 
    AttributeManager( pyVector, attrList_, "User-implemented Python vector class"),
    pyVector_(pyVector), has_ownership_(false) {
  }

  virtual ~PythonVector() {
    for( auto &m : method_ ) {
      Py_XDECREF( m.second.name );
    }
    if( has_ownership_ ) {
      Py_XDECREF( pyVector_ );
    }
  }

  int dimension( ) const { 

    if( method_["dimension"].impl ) {
      PyObject* pyDimension = PyObject_CallMethodObjArgs(pyVector_,method_["dimension"].name,NULL);   
      return static_cast<int>(PyLong_AsLong(pyDimension)); 
    } 
    else {
      return 0;
    }
  }  

  Teuchos::RCP<Vector> clone() const {
#ifdef PYROL_DEBUG_MODE
    std::cout << "PythonVector::clone()" << std::endl;
#endif
    PyObject* pyClone = PyObject_CallMethodObjArgs(pyVector_,method_["clone"].name,NULL);
    return Teuchos::rcp( new PythonVector( pyClone, true ) );
  }

  Teuchos::RCP<Vector> basis( const int i ) const {
    PyObject* pyBasis = PyObject_CallMethodObjArgs(pyVector_,method_["clone"].name,NULL);
    Teuchos::RCP<Vector> b = Teuchos::rcp( new PythonVector( pyBasis, true ) );
    b->zero();
    PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
    PyObject* pyOne = PyFloat_FromDouble(1.0);
    PyObject_CallMethodObjArgs(pyVector_,method_["__setitem__"].name,pyIndex,pyOne,NULL);
    Py_XDECREF(pyIndex);
    Py_XDECREF(pyOne);
    return b;
  }

  virtual const Vector & dual() const {
    if( method_["dual"].impl ) {
      PyObject *pyDual = PyObject_CallMethodObjArgs(pyVector_,method_["dual"].name,NULL);
      Teuchos::RCP<Vector> d = Teuchos::rcp( new PythonVector(pyDual,true) );
      return *d;
    }
    else {
      return *this;
    }
  }
  
  void applyUnary( const UnaryFunction &f ) {
    int dim = dimension();
    for(int i=0; i<dim; ++i) {
      setValue( i, f.apply( getValue(i) ) );
    }
  }

  void applyBinary( const BinaryFunction &f, const Vector &x ) {
    const PythonVector ex = Teuchos::dyn_cast<const PythonVector>(x);

    int dim = dimension();
      for(int i=0; i<dim; ++i) {
        setValue( i, f.apply( getValue(i), ex.getValue(i)) );
      }
  }

  double reduce( const ReductionOp &r ) const {
    double result = r.initialValue();
    int dim = dimension();
    for(int i=0; i<dim; ++i) {
      r.reduce(getValue(i),result); 
    }
    return result;
  }  

  void setValue (int i, double value) {
    PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
    PyObject* pyValue = PyFloat_FromDouble(value);
    PyObject_CallMethodObjArgs(pyVector_,method_["__setitem__"].name,pyIndex,pyValue,NULL);
    Py_DECREF(pyValue);
    Py_DECREF(pyIndex);
  }

  double getValue(int i) const {
    PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
    PyObject* pyValue = PyObject_CallMethodObjArgs(pyVector_,method_["__getitem__"].name,pyIndex,NULL);
    double value = PyFloat_AsDouble(pyValue);
    Py_DECREF(pyValue);
    Py_DECREF(pyIndex);
    return value; 
  }


}; // class PythonVector

} // namespace PyROL


#endif // PYROL_PYTHONVECTOR_HPP
