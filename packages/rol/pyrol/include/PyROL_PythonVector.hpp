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
#include "PyROL_BaseVector.hpp"

/** \class PyROL::PythonVector
 *  \brief Provides a ROL interface to generic vectors defined in Python
 *         which satisfy the interface requirements
 */

namespace PyROL {

class PythonVector : public BaseVector, public AttributeManager {

  using Vector          = ROL::Vector<double>;

  using UnaryFunction   = ROL::Elementwise::UnaryFunction<double>;
  using BinaryFunction  = ROL::Elementwise::BinaryFunction<double>;
  using ReductionOp     = ROL::Elementwise::ReductionOp<double>;

public:
  const static AttributeManager::Name className_;

private:
  const static AttributeManager::AttributeList attrList_;

  PyObject* pyVector_;

  bool has_ownership_;

public:

  PythonVector( PyObject* pyVector, bool has_ownership=false );
  PythonVector( const PythonVector & v );
  virtual ~PythonVector();
  int dimension() const;
  Teuchos::RCP<Vector> clone() const;
  Teuchos::RCP<Vector> basis( const int i ) const;
  virtual const Vector & dual() const;
  void plus( const Vector & x );
  void scale( const double alpha );
  double dot( const Vector &x ) const;
  double norm() const;
  void axpy( const double alpha, const Vector &x );
  void zero();
  void set( const Vector &x );
  void applyUnary( const UnaryFunction &f );
  void applyBinary( const BinaryFunction &f, const Vector &x );
  double reduce( const ReductionOp &r ) const;
  void setValue (int i, double value);
  double getValue(int i) const;
  PyObject* getPyVector(void);
  const PyObject* getPyVector(void) const;
  void print( std::ostream &os ) const;

}; // class PythonVector

} // namespace PyROL

// #include "PyROL_PythonVector_Impl.hpp"

#endif // PYROL_PYTHONVECTOR_HPP
