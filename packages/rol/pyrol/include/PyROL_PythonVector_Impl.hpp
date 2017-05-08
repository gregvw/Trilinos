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


#ifndef PYROL_PYTHONVECTOR_IMPL_HPP
#define PYROL_PYTHONVECTOR_IMPL_HPP

// #include "PyROL_PythonVector.hpp"
#include "PyROL_TypeConverters.hpp"

namespace PyROL {

PythonVector::PythonVector( PyObject* pyVector, bool has_ownership ) :
  AttributeManager( pyVector, attrList_, className_ ),
  pyVector_(pyVector), has_ownership_(has_ownership) {

  if(has_ownership_) {
    Py_INCREF(pyVector_);
  }
}

PythonVector::PythonVector( const PythonVector & v ):
  AttributeManager( v.pyVector_, attrList_, className_ ),
  pyVector_( v.pyVector_ ), has_ownership_(false) {
  std::cout << "PythonVector Copy Constructor" << std::endl;
//  Py_INCREF(pyVector_);
}


PythonVector::~PythonVector() {
  TEUCHOS_TEST_FOR_EXCEPTION( !(pyVector_->ob_refcnt), std::logic_error,
    "PythonVector() was called but pyVector already has zero references");
  int n = has_ownership_ ? 0 : 1;
  while(Py_REFCNT(pyVector_) > n)
    Py_DECREF(pyVector_);
 }

int PythonVector::dimension() const {
  if( method_["dimension"].impl ) {
    PyObject* pyDimension = PyObject_CallMethodObjArgs(pyVector_,method_["dimension"].name,NULL);
    TEUCHOS_TEST_FOR_EXCEPTION(!PyLong_Check(pyDimension), std::logic_error,
                               "dimension() returned incorrect type");
    return static_cast<int>(PyLong_AsLong(pyDimension));
  }
  else {
    return 0;
  }
}

Teuchos::RCP<ROL::Vector<double>>
PythonVector::clone() const {
  PyObject* pyClone = PyObject_CallMethodObjArgs(pyVector_,method_["clone"].name,NULL);
  Teuchos::RCP<ROL::Vector<double>> vclone = Teuchos::rcp( new PythonVector( pyClone, true ) );
  Py_INCREF(pyClone);
  return vclone;
}

Teuchos::RCP<ROL::Vector<double>>
PythonVector::basis( const int i ) const {
  PyObject* pyBasis = PyObject_CallMethodObjArgs(pyVector_,method_["clone"].name,NULL);
  Teuchos::RCP<ROL::Vector<double>> b = Teuchos::rcp( new PythonVector( pyBasis, true ) );
  Py_DECREF(pyBasis);
  b->zero();
  PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
  PyObject* pyOne = PyFloat_FromDouble(1.0);
  PyObject_CallMethodObjArgs(pyVector_,method_["__setitem__"].name,pyIndex,pyOne,NULL);
  Py_DECREF(pyIndex);
  Py_DECREF(pyOne);
  return b;
}

const ROL::Vector<double> & PythonVector::dual() const {
  if( method_["dual"].impl ) {
    PyObject *pyDual = PyObject_CallMethodObjArgs(pyVector_,method_["dual"].name,NULL);
    Teuchos::RCP<Vector> d = Teuchos::rcp( new PythonVector(pyDual,true) );
    return *d;
  }
  else {
    return *this;
  }
}

void PythonVector::plus( const ROL::Vector<double> & x ) {
  if( method_["plus"].impl ) {
    // Borrowed reference
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject_CallMethodObjArgs(pyVector_,method_["plus"].name,pyX,NULL);
  }
  else {
    this->applyBinary(ROL::Elementwise::Plus<double>(),x);
  }
}

void PythonVector::scale( const double alpha ) {
  if( method_["scale"].impl ) {
    PyObject* pyAlpha = PyFloat_FromDouble(alpha);
    PyObject_CallMethodObjArgs(pyVector_,method_["scale"].name,pyAlpha,NULL);
    Py_DECREF(pyAlpha);
  }
  else {
    this->applyUnary(ROL::Elementwise::Scale<double>(alpha));
  }
}

double PythonVector::dot( const ROL::Vector<double> &x ) const {
  double value = 0;
  if( method_["dot"].impl ) {
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject* pyValue;
    // This is supposed to return a new reference
    pyValue = PyObject_CallMethodObjArgs(pyVector_,method_["dot"].name,pyX,NULL);
    TEUCHOS_TEST_FOR_EXCEPTION(!PyFloat_Check(pyValue), std::logic_error,
                               "dot() returned incorrect type");
    value = PyFloat_AsDouble(pyValue);
    Py_DECREF(pyValue);// causes seg fault
    // Py_XDECREF(pyValue);
  }
  else {
    const PythonVector ex = Teuchos::dyn_cast<const PythonVector>(x);
    int dim = dimension();
    for( int i=0; i<dim; ++i ) {
      value += getValue(i)*ex.getValue(i);
    }
  }
  return value;
}

double PythonVector::norm( ) const {
  double value = 0;
  if( method_["norm"].impl ) {
    PyObject* pyValue;
    pyValue = PyObject_CallMethodObjArgs(pyVector_,method_["norm"].name,NULL);

    TEUCHOS_TEST_FOR_EXCEPTION(!PyFloat_Check(pyValue), std::logic_error,
                               "norm() returned incorrect type");
    value = PyFloat_AsDouble(pyValue);
    Py_DECREF(pyValue);
  }
  else {
    int dim = dimension();
    for( int i=0; i<dim; ++i ) {
      value += getValue(i)*getValue(i);
    }
    value = std::sqrt(value);
  }
  return value;
}

void PythonVector::axpy( const double alpha, const ROL::Vector<double> &x ) {
  if( method_["axpy"].impl ) {
    PyObject* pyAlpha = PyFloat_FromDouble(alpha);
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject_CallMethodObjArgs(pyVector_,method_["axpy"].name,pyAlpha,pyX,NULL);
    Py_DECREF(pyAlpha);
  }
  else {
    const PythonVector ex = Teuchos::dyn_cast<const PythonVector>(x);
    int dim = dimension();
    for( int i=0; i<dim; ++i ) {
      setValue(i, getValue(i) + alpha*ex.getValue(i));
    }
  }
}

void PythonVector::zero( ) {
  if( method_["zero"].impl ) {
     PyObject_CallMethodObjArgs(pyVector_,method_["zero"].name,NULL);
  }
  else {
    int dim = dimension();
    for( int i=0; i<dim; ++i ) {
      setValue(i, 0.0);
    }
  }
}

void PythonVector::set( const ROL::Vector<double> &x ) {
  if( method_["set"].impl ) {
    const PyObject* pyX = PyObject_FromVector(x);
    PyObject_CallMethodObjArgs(pyVector_,method_["set"].name,pyX,NULL);
  }
  else {
    const PythonVector ex = Teuchos::dyn_cast<const PythonVector>(x);
    int dim = dimension();
    for( int i=0; i<dim; ++i ) {
      setValue(i, ex.getValue(i));
    }
  }
}

void PythonVector::applyUnary( const UnaryFunction &f ) {

  // Try to get all entries at once from buffer protocol object (e.g. numpy array)

  PyObject* pySlice = PySlice_New(NULL, NULL, NULL);
  PyObject* pyArray = PyObject_CallMethodObjArgs(pyVector_,method_["__getitem__"].name, pySlice, NULL);
  // If __getitem__ returns a Py_buffer protocol object when given a PySlice, then go ahead
  if (PyObject_CheckBuffer(pyArray)) {
    Py_buffer view;
    PyObject_GetBuffer(pyArray, &view, 0);
    const double *data = static_cast<double*>(view.buf);
    int dim = dimension();
    for(int i = 0; i < dim; ++i) {
      setValue(i, f.apply(data[i]));
    }
    PyBuffer_Release(&view);
  }
  else {
    int dim = dimension();
    for(int i=0; i<dim; ++i) {
      setValue( i, f.apply( getValue(i) ) );
    }
  }

}

void PythonVector::applyBinary( const BinaryFunction &f, const ROL::Vector<double> &x ) {
  const PythonVector ex = Teuchos::dyn_cast<const PythonVector>(x);

  int dim = dimension();

  // Try to get all entries at once from buffer protocol object (e.g. numpy array)
  PyObject* pySlice = PySlice_New(NULL, NULL, NULL);
  PyObject* pyArraySelf = PyObject_CallMethodObjArgs(pyVector_,method_["__getitem__"].name, pySlice, NULL);
  PyObject* pyArrayX = PyObject_CallMethodObjArgs(ex.pyVector_,method_["__getitem__"].name, pySlice, NULL);

  if (PyObject_CheckBuffer(pyArraySelf) and PyObject_CheckBuffer(pyArrayX)) {

    Py_buffer self_view;
    PyObject_GetBuffer(pyArraySelf, &self_view, 0);
    const double *self_data = static_cast<double*>(self_view.buf);

    Py_buffer x_view;
    PyObject_GetBuffer(pyArrayX, &x_view, 0);
    const double *x_data = static_cast<double*>(x_view.buf);

    for(int i = 0; i < dim; ++i) {
      setValue( i, f.apply( self_data[i], x_data[i]) );
    }

    PyBuffer_Release(&self_view);
    PyBuffer_Release(&x_view);
  }
  else {
    for(int i=0; i<dim; ++i) {
      setValue( i, f.apply( getValue(i), ex.getValue(i)) );
    }
  }

}

double PythonVector::reduce( const ReductionOp &r ) const {
  double result = r.initialValue();

  PyObject* pySlice = PySlice_New(NULL, NULL, NULL);
  PyObject* pyArray = PyObject_CallMethodObjArgs(pyVector_,method_["__getitem__"].name, pySlice, NULL);
  // If __getitem__ returns a Py_buffer protocol object when given a PySlice, then go ahead
  if (PyObject_CheckBuffer(pyArray)) {
    Py_buffer view;
    PyObject_GetBuffer(pyArray, &view, 0);
    const double *data = static_cast<double*>(view.buf);
    int dim = dimension();
    for(int i = 0; i < dim; ++i) {
      r.reduce(data[i],result);
    }
    PyBuffer_Release(&view);
  }
  else {
    int dim = dimension();
    for(int i=0; i<dim; ++i) {
      r.reduce(getValue(i),result);
    }
  }

  return result;
}

void PythonVector::setValue (int i, double value) {
  PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
  PyObject* pyValue = PyFloat_FromDouble(value);
  PyObject_CallMethodObjArgs(pyVector_,method_["__setitem__"].name,pyIndex,pyValue,NULL);
  Py_DECREF(pyValue);
  Py_DECREF(pyIndex);
}

double PythonVector::getValue(int i) const {
  PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
  PyObject* pyValue = PyObject_CallMethodObjArgs(pyVector_,method_["__getitem__"].name,pyIndex,NULL);

  TEUCHOS_TEST_FOR_EXCEPTION(!PyFloat_Check(pyValue), std::logic_error,
    "__getitem__ returned incorrect type");

  double value = PyFloat_AsDouble(pyValue);
  Py_DECREF(pyIndex);
  return value;
}

PyObject* PythonVector::getPyVector(void) {
  return pyVector_;
}

const PyObject* PythonVector::getPyVector(void) const {
  return pyVector_;
}

void PythonVector::print( std::ostream &os ) const {
  for( int i=0; i<dimension(); ++ i ) {
    os << getValue(i) << "  ";
  }
  os << std::endl;
}

} // namespace PyROL


#endif // PYROL_PYTHONVECTOR_IMPL_HPP
