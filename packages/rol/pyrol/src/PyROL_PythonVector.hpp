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

#include "PyROL.hpp"

#include <array>
#include <ostream>
#include <string>
#include <tuple>

/** \class PyROL::PythonVector
 *  \brief Provides a ROL interface to generic vectors defined in Python 
 *         which satisfy the interface requirements
 */

namespace PyROL {

class PythonVector : public ROL::Vector<double> {

  using Required    = bool;
  using Implemented = bool;
  using Name        = std::string;
  using Attribute   = std::tuple<Name,Required>;
  using Method      = std::tuple<PyObject*,Required,Implemented>;

  using Vector          = ROL::Vector<double>;

  using UnaryFunction   = ROL::Elementwise::UnaryFunction<double>;  
  using BinaryFunction  = ROL::Elementwise::BinaryFunction<double>;  
  using ReductionOp     = ROL::Elementwise::ReductionOp<double>;  


private: 
  
  static const int numAttr_       = 19;

  static const int NAME           = 0;
  static const int IS_REQUIRED    = 1;
  static const int IS_IMPLEMENTED = 2;

  const static std::array<Attribute, numAttr_> attribute_{ { 
     //               method name, required
     std::make_tuple( "plus",        true    ),
     std::make_tuple( "scale",       true    ), 
     std::make_tuple( "dot",         true    ), 
     std::make_tuple( "norm",        true    ),
     std::make_tuple( "clone",       true    ),
     std::make_tuple( "getVector",   true    ),
     std::make_tuple( "axpy",        false   ),
     std::make_tuple( "zero",        false   ),
     std::make_tuple( "basis",       false   ),
     std::make_tuple( "dimension",   false   ), 
     std::make_tuple( "set",         false   ),
     std::make_tuple( "dual",        false   ),
     std::make_tuple( "applyUnary",  false   ),
     std::make_tuple( "applyBinary", false   ),
     std::make_tuple( "reduce",      false   ),
     std::make_tuple( "print",       false   ), 
     std::make_tuple( "checkVector", false   ),
     std::make_tuple( "__getitem__", false   ),  // Overload the [] operator for NumPy-like access
     std::make_tuple( "__setitem__", false   ),  // Overload the [] operator for NumPy-like access
  } };

  std::map<std::string,Method> method_;

  PyObject* pyVector_;

  bool has_owenership;

  std::string pyMethod( const Method* m ) const {
    return std::get<NAME>(m);
  }

  bool is_implemented( const Method* m ) const {
    return std::get<IS_IMPLEMENTED>(m);
  }

public:

  PythonVector( PyObject* pyVector, bool has_ownership=false ) : 
    pyVector_(pyVector), has_ownership_(false) {

    // Loop over attributes - determine which are implemented
    for( auto a : attribute_ ) {

      Name        name = std::get<NAME>(a);
      Required    req  = std::get<IS_REQUIRED>(a);
      Implemented impl = PyObject_HasAttr( pyVector_, Name );

      TEUCHOS_TEST_FOR_EXCEPTION( req & !impl, std::logic_error, 
        "Error: The Python vector class must implement a method " << name );

      method_[name] = std::make_tuple( PyString_FromString(name), req, impl );
    }
  }

  virtual ~PythonVector() {
    for( auto &m : method_ ) {
      Py_DECREF( pyMethod(m) );
    }
    if( has_ownership_ ) {
      Py_DECREF( pyVector_ );
    }
  }

  // Optional
  int dimension( ) const { 
    Method dim = method_["dimension"];

    if( is_implemented(dim) ) {
      PyObject* pyDimension = PyObject_CallMethodObjArgs(pyVector_,pyMethod(dim),NULL);   
      return static_cast<int>(PyLong_AsLong(pyDimension)); 
    } 
    else {
      return 0;
    }
  }  

  // Required
  void plus( const Vector &x ) { 
    Method add = method_["plus"];
    const PyObject* pyX = Teuchos::dyn_cast<PythonVector>(x).getVector();
    PyObject_CallMethodObjArgs(pyVector_,pyMethod(add),pyX,NULL);
  }
  
  // Required
  void scale( const double alpha ) {
    Method scalarMultiply = method_["scale"];
    PyObject* pyAlpha = PyFloat_FromDouble(alpha);  
    PyObject_CallMethodObjArgs(pyVector_,pyMethod(scalarMultiply),NULL); 
    Py_DECREF(pyAlpha);
  }

  // Required
  virtual double dot( const Vector &x ) {
    Method innerProduct = method_["dot"];
    const PyObject* pyX = Teuchos::dyn_cast<PythonVector>(x).getVector();
    PyObject* pyResult = PyObject_CallMethodObjArgs(pyVector_,pyMethod(innerProduct),pyX,NULL); 
    double result = PyFloat_AsDouble(pyResult);
    Py_DECREF(pyResult);
    return result;
  }

  // Required
  void norm( ) const {
    Method vectorNorm = method_["norm"];
    PyObject* pyResult = PyObject_CallMethodObjArgs(pyVector_,pyMethod(vectorNorm),NULL); 
    double result = PyFloat_AsDouble(pyResult);
    Py_DECREF(pyResult);
    return result;
  }

  // Required
  Teuchos::RCP<Vector> clone() const {
    Method create = method_["clone"];
    PyObject* pyClone = PyObject_CallMethodObjArgs(pyVector_,pyMethod(create),NULL);
    return Teuchos::rcp( new PythonVector( PyObject, true ) );
  }

  // Optional
  void axpy( const Real alpha, const Vector &x ) {
    Method addMultiple = method_["axpy"];

    if( is_implemented(addMultiple) ) {
      PyObject* pyAlpha = PyFloat_FromDouble(alpha);  
      const PyObject* pyX = Teuchos::dyn_cast<PythonVector>(x).getVector();
      PyObject_CallMethodObjArgs(pyVector_,pyMethod(addMultiple),pyAlpha,pyX,NULL);
      Py_DECREF(pyAlpha);
    }
    else {
      Teuchos::RCP<Vector> ax = x.clone();
      ax->set(x);
      ax->scale(alpha);
      this->plus(*ax);
    }
  }

  // Optional
  void zero( ) { 
    Method fillZeros = method_["zero"];
   
    if( is_implemented(fillZeros) ) {
      PyObject_CallMethodObjArgs(pyVector_,pyMethod(fillZeros),NULL);  
    }
    else {
      this->scale(0.0);
    }
  }

  // Optional
  Teuchos::RCP<Vector> basis( const int i ) const {
    Method canonicalVector = method_["basis"];
    Method setItem = method_["__setitem__"];

    Teuchos::RCP<Vector> b;

    if( is_implemented(canonicalVector) ) {
      PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
      PyObject* pyB = PyObject_CallMethodObjArgs(pyVector_,pyMethod(canonicalVector),pyIndex,NULL);
      b = Teuchos::rcp( new PythonVector( pyB, true ) );
      Py_DECREF(pyIndex);
    }
    else if( is_implemented( setItem ) ) {
      PyObject* pyIndex = PyLong_FromLong(static_cast<long>(i));
      PyObject* pyOne = PyLong_FromLong(1l);  
      b = this->clone();
      b->zero();
      pyObject* pyB = b->getVector();
      PyObject_CallMethodObjArgs(pyVector_,pyMethod(setItem),pyIndex,pyOne,NULL);
      Py_DECREF(pyIndex);
      Py_DECREF(pyOne);
    }
    else {
      b = Teuchos::null;
    }   
    return b;
  }

  // Optional
  void set( const Vector<Real> & x ) {
    Method setVector = method["set"];
    if( is_implemented(setVector) ) {
      PyObject* pyX = Teuchos::dyn_cast<PythonVector>(x).getVector();  
      PyObject_CallMethodObjArgs(pyVector_,pyMethod(setVector),pyX,NULL);
    } 
    else {
      this->zero();
      this->plus(x);  
    }
  }

  // Optional
  virrtual const Vector & dual() const {
    Method dualVector = method["dual"];
    if( is_implemented(dualVector) ) {
      PyObject *pyDual = PyObject_CallMethodObjArgs(pyVector_,pyMethod(dualVector),NULL);
      Teuchos::RCP<Vector> d = Teuchos::rcp( new PythonVector(pyDual,true) );
      return *d;
    }
    else {
      return *this;
    }
  }
  
  // Optional
  void applyUnary( const UnaryFunction<Real> &f ) {
        
  }

  // Optional
  void applyBinary( const BinaryFunction<Real> &f, const Vector &x ) {

  }

  // Optional
  double reduce( const ReductionOp &r ) const {

  }  

  PyObject* getVector() {
    return pyVector_;
  }

  const PyObject* getVector() const {
    return pyVector_;
  }


}; // class PythonVector

} // namespace PyROL


#endif // PYROL_PYTHONVECTOR_HPP
