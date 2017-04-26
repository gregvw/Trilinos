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


#ifndef PYROL_BASEVECTOR_HPP
#define PYROL_BASEVECTOR_HPP

#include "PyROL.hpp"

namespace PyROL {

/** \class BaseVector provides an abstract interface like ROL::Vector, but with
           the ability to return the member PyObject* vector. This enables
           Objectives, Constraints, etc to be vector type agnostic */

class BaseVector : public virtual ROL::ElementwiseVector<double> {

  using Vector = ROL::Vector<double>;

  using UnaryFunction   = ROL::Elementwise::UnaryFunction<double>;  
  using BinaryFunction  = ROL::Elementwise::BinaryFunction<double>;  
  using ReductionOp     = ROL::Elementwise::ReductionOp<double>;  

public:

  virtual ~BaseVector() {}

  // All PyROL vectors must implement these two methods since
  // every derived class contains a PyObject*  
  virtual PyObject* getPyVector() = 0;
  virtual const PyObject* getPyVector() const = 0;


  virtual Teuchos::RCP<Vector> clone() const = 0;

  virtual Teuchos::RCP<Vector> basis( const int i ) const {return Teuchos::null;}

  virtual int dimension() const {return 0;}

  virtual void applyUnary( const UnaryFunction &f ) {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error,
      "The method applyUnary wass called, but not implemented" << std::endl); 
  }

  virtual void applyBinary( const BinaryFunction &f, const Vector &x ) {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error,
      "The method applyBinary wass called, but not implemented" << std::endl); 
  }

  virtual double reduce( const ReductionOp &r ) const {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::logic_error,
      "The method reduce was called, but not implemented" << std::endl); 
  }

  virtual void print( std::ostream &outStream ) const {
    outStream << "The method print was called, but not implemented" << std::endl;
  }    

}; // class BaseVector

} // namespace PyROL



#endif // PYROL_BASEVECTOR_HPP

