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

#include "ROL_Vector.hpp"

#ifndef ROL_PARTITIONED_VECTOR_H
#define ROL_PARTITIONED_VECTOR_H

/** @ingroup la_group
 *  \class ROL::PartitionedVector
 *  \brief Defines the linear algebra of vector space on a generic partitioned vector
 */


namespace ROL {

template<class Real>
class PartitionedVector : public Vector<Real> {

  typedef Vector<Real>                  V;
  typedef std::shared_ptr<V>            Vptr;
  typedef PartitionedVector<Real>       PV;

private:
  const std::vector<Vptr>                    vecs_;
  mutable std::vector<Vptr>             dual_vecs_;
  mutable std::shared_ptr<PV>              dual_pvec_;
public:

  typedef typename std::vector<PV>::size_type    size_type;

  PartitionedVector( const std::vector<Vptr> &vecs ) : vecs_(vecs) {
    for( size_type i=0; i<vecs_.size(); ++i ) {
      dual_vecs_.push_back((vecs_[i]->dual()).clone());
    }
  }

  void set( const V &x ) {

    const PV &xs = dynamic_cast<const PV&>(dynamic_cast<const V&>(x));

    TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->set(*xs.get(i));
    }
  }

  void plus( const V &x ) {

    const PV &xs = dynamic_cast<const PV&>(dynamic_cast<const V&>(x));

    TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->plus(*xs.get(i));
    }
  }

  void scale( const Real alpha ) {
    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->scale(alpha);
    }
  }

  void axpy( const Real alpha, const V &x ) {

    const PV &xs = dynamic_cast<const PV&>(x);

    TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->axpy(alpha,*xs.get(i));
    }
  }

  Real dot( const V &x ) const {

    const PV &xs = dynamic_cast<const PV&>(x);

   TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    Real result = 0;
    for( size_type i=0; i<vecs_.size(); ++i ) {
      result += vecs_[i]->dot(*xs.get(i));
    }
    return result;
  }

  Real norm() const {
    Real result = 0;
    for( size_type i=0; i<vecs_.size(); ++i ) {
      result += std::pow(vecs_[i]->norm(),2);
    }
    return std::sqrt(result);
  }

  Vptr clone() const {
    std::vector<Vptr> clonevec;
    for( size_type i=0; i<vecs_.size(); ++i ) {
      clonevec.push_back(vecs_[i]->clone());
    }
    return std::make_shared<PV>(clonevec);
  }

  const V& dual(void) const {
    for( size_type i=0; i<vecs_.size(); ++i ) {
      dual_vecs_[i]->set(vecs_[i]->dual());
    }
    dual_pvec_ = std::make_shared<PV>( dual_vecs_ );
    return *dual_pvec_;
  }

  Vptr basis( const int i ) const {

    TEUCHOS_TEST_FOR_EXCEPTION( i >= dimension() || i<0,
                                std::invalid_argument,
                                "Error: Basis index must be between 0 and vector dimension." );

    Vptr bvec = clone();

    // Downcast
    PV &eb = dynamic_cast<PV&>(*bvec);

    int begin = 0;
    int end = 0;

    // Iterate over subvectors
    for( size_type j=0; j<vecs_.size(); ++j ) {

      end += vecs_[j]->dimension();

      if( begin<= i && i<end ) {
        eb.set(j, *(vecs_[j]->basis(i-begin)) );
      }
      else {
        eb.zero(j);
      }

      begin = end;

    }
    return bvec;
  }

  int dimension() const {
    int total_dim = 0;
    for( size_type j=0; j<vecs_.size(); ++j ) {
      total_dim += vecs_[j]->dimension();
    }
    return total_dim;
  }

  void zero() {
    for( size_type j=0; j<vecs_.size(); ++j ) {
      vecs_[j]->zero();
    }
  }

  // Apply the same unary function to each subvector
  void applyUnary( const Elementwise::UnaryFunction<Real> &f ) {
    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->applyUnary(f);
    }
  }

  // Apply the same binary function to each pair of subvectors in this vector and x
  void applyBinary( const Elementwise::BinaryFunction<Real> &f, const V &x ) {
    const PV &xs = dynamic_cast<const PV&>(x);

    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->applyBinary(f,*xs.get(i));
    }
  }

  Real reduce( const Elementwise::ReductionOp<Real> &r ) const {
    Real result = r.initialValue();

    for( size_type i=0; i<vecs_.size(); ++i ) {
      r.reduce(vecs_[i]->reduce(r),result);
    }
    return result;
  }

  void print( std::ostream &outStream ) const {
    for( size_type i=0; i<vecs_.size(); ++i ) {
      outStream << "V[" << i << "]: ";
      vecs_[i]->print(outStream);
    }
  }

  // Methods that do not exist in the base class

  std::shared_ptr<const Vector<Real> > get(size_type i) const {
    return vecs_[i];
  }

  std::shared_ptr<Vector<Real> > get(size_type i) {
    return vecs_[i];
  }

  void set(size_type i, const V &x) {
    vecs_[i]->set(x);
  }

  void zero(size_type i) {
    vecs_[i]->zero();
  }

  size_type numVectors() const {
    return vecs_.size();
  }

};

// Helper methods
template<class Real>
std::shared_ptr<Vector<Real> > CreatePartitionedVector( const std::shared_ptr<Vector<Real> > &a ) {


  typedef std::shared_ptr<Vector<Real> >       Vptr;
  typedef PartitionedVector<Real>  PV;

  Vptr temp[] = {a};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+1));
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreatePartitionedVector( const std::shared_ptr<const Vector<Real> > &a ) {


  typedef std::shared_ptr<const Vector<Real> >      Vptr;
  typedef const PartitionedVector<Real> PV;

  Vptr temp[] = {a};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+1));
}

template<class Real>
std::shared_ptr<Vector<Real> > CreatePartitionedVector( const std::shared_ptr<Vector<Real> > &a,
                                                     const std::shared_ptr<Vector<Real> > &b ) {


  typedef std::shared_ptr<Vector<Real> >      Vptr;
  typedef PartitionedVector<Real> PV;

  Vptr temp[] = {a,b};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+2));
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreatePartitionedVector( const std::shared_ptr<const Vector<Real> > &a,
                                                           const std::shared_ptr<const Vector<Real> > &b ) {


  typedef std::shared_ptr<const Vector<Real> >      Vptr;
  typedef const PartitionedVector<Real> PV;

  Vptr temp[] = {a,b};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+2));
}

template<class Real>
std::shared_ptr<Vector<Real> > CreatePartitionedVector( const std::shared_ptr<Vector<Real> > &a,
                                                     const std::shared_ptr<Vector<Real> > &b,
                                                     const std::shared_ptr<Vector<Real> > &c ) {


  typedef std::shared_ptr<Vector<Real> >      Vptr;
  typedef PartitionedVector<Real> PV;

  Vptr temp[] = {a,b,c};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+3));
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreatePartitionedVector( const std::shared_ptr<const Vector<Real> > &a,
                                                           const std::shared_ptr<const Vector<Real> > &b,
                                                           const std::shared_ptr<const Vector<Real> > &c ) {


  typedef std::shared_ptr<const Vector<Real> >      Vptr;
  typedef const PartitionedVector<Real> PV;

  Vptr temp[] = {a,b,c};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+3));
}

template<class Real>
std::shared_ptr<Vector<Real> > CreatePartitionedVector( const std::shared_ptr<Vector<Real> > &a,
                                                     const std::shared_ptr<Vector<Real> > &b,
                                                     const std::shared_ptr<Vector<Real> > &c,
                                                     const std::shared_ptr<Vector<Real> > &d ) {


  typedef std::shared_ptr<Vector<Real> >      Vptr;
  typedef PartitionedVector<Real> PV;

  Vptr temp[] = {a,b,c,d};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+4));
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreatePartitionedVector( const std::shared_ptr<const Vector<Real> > &a,
                                                           const std::shared_ptr<const Vector<Real> > &b,
                                                           const std::shared_ptr<const Vector<Real> > &c,
                                                           const std::shared_ptr<const Vector<Real> > &d ) {


  typedef std::shared_ptr<const Vector<Real> >      Vptr;
  typedef const PartitionedVector<Real> PV;

  Vptr temp[] = {a,b,c,d};
  return std::make_shared<PV>(std::vector<Vptr>(temp, temp+4));
}

} // namespace ROL

#endif // ROL_PARTITIONED_VECTOR_H
