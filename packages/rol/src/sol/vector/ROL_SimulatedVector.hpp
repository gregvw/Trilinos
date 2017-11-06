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
#include "ROL_SampleGenerator.hpp"

#ifndef ROL_SIMULATED_VECTOR_H
#define ROL_SIMULATED_VECTOR_H

/** @ingroup la_group
 *  \class ROL::SimulatedVector
 *  \brief Defines the linear algebra of a vector space on a generic
           partitioned vector where the individual vectors are
           distributed in batches defined by ROL::BatchManager.
           This is a batch-distributed version of ROL::PartitionedVector.
 */


namespace ROL {

template<class Real>
class PrimalSimulatedVector;

template<class Real>
class DualSimulatedVector;

template<class Real>
class SimulatedVector : public Vector<Real> {

  typedef Vector<Real>                       V;
  typedef std::shared_ptr<V>                    std::shared_ptrV;
  typedef std::shared_ptr<BatchManager<Real> >  std::shared_ptrBM;
  typedef SimulatedVector<Real>              PV;

private:
  const std::vector<std::shared_ptrV>                    vecs_;
  std::shared_ptr<BatchManager<Real> >          bman_;
  mutable std::vector<std::shared_ptrV>             dual_vecs_;
  mutable std::shared_ptr<PV>              dual_pvec_;
public:

  typedef typename std::vector<PV>::size_type    size_type;

  SimulatedVector( const std::vector<std::shared_ptrV> &vecs, const std::shared_ptrBM &bman ) : vecs_(vecs), bman_(bman) {
    for( size_type i=0; i<vecs_.size(); ++i ) {
      dual_vecs_.push_back((vecs_[i]->dual()).clone());
    }
  }

  void set( const V &x ) {
    
    const PV &xs = dynamic_cast<const PV>(dynamic_cast<const V>(x));

    TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->set(*xs.get(i));
    }
  }

  void plus( const V &x ) {
    
    const PV &xs = dynamic_cast<const PV>(dynamic_cast<const V>(x));

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
    
    const PV &xs = dynamic_cast<const PV>(x);

    TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    for( size_type i=0; i<vecs_.size(); ++i ) {
      vecs_[i]->axpy(alpha,*xs.get(i));
    }
  }

  virtual Real dot( const V &x ) const {
    
    const PV &xs = dynamic_cast<const PV>(x);

   TEUCHOS_TEST_FOR_EXCEPTION( numVectors() != xs.numVectors(),
                                std::invalid_argument,
                                "Error: Vectors must have the same number of subvectors." );

    Real locresult = 0;
    Real result = 0;
    for( size_type i=0; i<vecs_.size(); ++i ) {
      locresult += vecs_[i]->dot(*xs.get(i));
    }

    bman_->sumAll(&locresult, &result, 1);

    return result;
  }

  Real norm() const {
    return std::sqrt(dot(*this));
  }

  virtual std::shared_ptrV clone() const {
    
    

    std::vector<std::shared_ptrV> clonevec;
    for( size_type i=0; i<vecs_.size(); ++i ) {
      clonevec.push_back(vecs_[i]->clone());
    }
    return std::make_shared<PV>(clonevec, bman_);
  }

  virtual const V& dual(void) const {
    

    for( size_type i=0; i<vecs_.size(); ++i ) {
      dual_vecs_[i]->set(vecs_[i]->dual());
    }
    dual_pvec_ = std::make_shared<PV>( dual_vecs_, bman_ );
    return *dual_pvec_;
  }

  std::shared_ptrV basis( const int i ) const { // this must be fixed for distributed batching

    TEUCHOS_TEST_FOR_EXCEPTION( i >= dimension() || i<0,
                                std::invalid_argument,
                                "Error: Basis index must be between 0 and vector dimension." );

    
    
    

    std::shared_ptrV bvec = clone();

    // Downcast
    PV &eb = dynamic_cast<PV>(*bvec);

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

  int dimension() const { // this must be fixed for distributed batching
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
    const PV &xs = dynamic_cast<const PV>(x);

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

  // Methods that do not exist in the base class

  // In distributed batching mode, these are understood to take local indices.

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
std::shared_ptr<Vector<Real> > CreateSimulatedVector( const std::shared_ptr<Vector<Real> > &a, const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<Vector<Real> >       std::shared_ptrV;
  typedef SimulatedVector<Real>  PV;

  std::shared_ptrV temp[] = {a};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+1), bman );
}

template<class Real>
class PrimalSimulatedVector : public SimulatedVector<Real> {
private:
  const std::vector<std::shared_ptr<Vector<Real> > >   vecs_;
  const std::shared_ptr<BatchManager<Real> >           bman_;
  const std::shared_ptr<SampleGenerator<Real> >        sampler_;
  mutable std::vector<std::shared_ptr<Vector<Real> > > dual_vecs_;
  mutable std::shared_ptr<DualSimulatedVector<Real> >  dual_pvec_;
  mutable bool isDualInitialized_;
public:

  PrimalSimulatedVector(const std::vector<std::shared_ptr<Vector<Real> > > &vecs,
                        const std::shared_ptr<BatchManager<Real> >         &bman,
                        const std::shared_ptr<SampleGenerator<Real> >      &sampler)
    : SimulatedVector<Real>(vecs,bman), vecs_(vecs), bman_(bman), sampler_(sampler),
      isDualInitialized_(false) {
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      dual_vecs_.push_back((vecs_[i]->dual()).clone());
    }
  }

  Real dot(const Vector<Real> &x) const {
    const SimulatedVector<Real> &xs
      = dynamic_cast<const SimulatedVector<Real> >(x);

   TEUCHOS_TEST_FOR_EXCEPTION( sampler_->numMySamples() != static_cast<int>(xs.numVectors()),
                               std::invalid_argument,
                               "Error: Vectors must have the same number of subvectors." );

    Real c = 0;
    Real locresult = 0;
    Real result = 0;
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      //locresult += sampler_->getMyWeight(i) * vecs_[i]->dot(*xs.get(i));
      Real y = sampler_->getMyWeight(i) * vecs_[i]->dot(*xs.get(i)) - c;
      Real t = locresult + y;
      c = (t - locresult) - y;
      locresult = t;
    }

    bman_->sumAll(&locresult, &result, 1);

    return result;
  }

  std::shared_ptr<Vector<Real> > clone(void) const {
    std::vector<std::shared_ptr<Vector<Real> > > clonevec;
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      clonevec.push_back(vecs_[i]->clone());
    }
    return std::make_shared<PrimalSimulatedVector<Real>>(clonevec, bman_, sampler_);
  }

  const Vector<Real>& dual(void) const {
    if (!isDualInitialized_) {
      dual_pvec_ = std::make_shared<DualSimulatedVector<Real>>(dual_vecs_, bman_, sampler_);
      isDualInitialized_ = true;
    }
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      dual_vecs_[i]->set(vecs_[i]->dual());
      dual_vecs_[i]->scale(sampler_->getMyWeight(i));
    }
    return *dual_pvec_;
  }

};

template<class Real>
class DualSimulatedVector : public SimulatedVector<Real> {
private:
  const std::vector<std::shared_ptr<Vector<Real> > >    vecs_;
  const std::shared_ptr<BatchManager<Real> >            bman_;
  const std::shared_ptr<SampleGenerator<Real> >         sampler_;
  mutable std::vector<std::shared_ptr<Vector<Real> > >  primal_vecs_;
  mutable std::shared_ptr<PrimalSimulatedVector<Real> > primal_pvec_;
  mutable bool isPrimalInitialized_;
public:

  DualSimulatedVector(const std::vector<std::shared_ptr<Vector<Real> > > &vecs,
                      const std::shared_ptr<BatchManager<Real> >         &bman,
                      const std::shared_ptr<SampleGenerator<Real> >      &sampler)
    : SimulatedVector<Real>(vecs,bman), vecs_(vecs), bman_(bman), sampler_(sampler),
      isPrimalInitialized_(false) {
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      primal_vecs_.push_back((vecs_[i]->dual()).clone());
    }
  }

  Real dot(const Vector<Real> &x) const {
    const SimulatedVector<Real> &xs
      = dynamic_cast<const SimulatedVector<Real> >(x);

   TEUCHOS_TEST_FOR_EXCEPTION( sampler_->numMySamples() != static_cast<Real>(xs.numVectors()),
                               std::invalid_argument,
                               "Error: Vectors must have the same number of subvectors." );

    Real c = 0;
    Real locresult = 0;
    Real result = 0;
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      //locresult += vecs_[i]->dot(*xs.get(i)) / sampler_->getMyWeight(i);
      Real y = vecs_[i]->dot(*xs.get(i)) / sampler_->getMyWeight(i) - c;
      Real t = locresult + y;
      c = (t - locresult) - y;
      locresult = t;
    }

    bman_->sumAll(&locresult, &result, 1);

    return result;
  }

  std::shared_ptr<Vector<Real> > clone(void) const {
    std::vector<std::shared_ptr<Vector<Real> > > clonevec;
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      clonevec.push_back(vecs_[i]->clone());
    }
    return std::make_shared<DualSimulatedVector<Real>>(clonevec, bman_, sampler_);
  }

  const Vector<Real>& dual(void) const {
    if (!isPrimalInitialized_) {
      primal_pvec_ = std::make_shared<PrimalSimulatedVector<Real>>(primal_vecs_, bman_, sampler_);
      isPrimalInitialized_ = true;
    }
    const Real one(1);
    for( int i=0; i<sampler_->numMySamples(); ++i ) {
      primal_vecs_[i]->set(vecs_[i]->dual());
      primal_vecs_[i]->scale(one/sampler_->getMyWeight(i));
    }
    return *primal_pvec_;
  }

};

template<class Real>
std::shared_ptr<const Vector<Real> > CreateSimulatedVector( const std::shared_ptr<const Vector<Real> > &a, const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<const Vector<Real> >      std::shared_ptrV;
  typedef const SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+1), bman );
}

template<class Real>
std::shared_ptr<Vector<Real> > CreateSimulatedVector( const std::shared_ptr<Vector<Real> > &a,
                                                   const std::shared_ptr<Vector<Real> > &b,
                                                   const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<Vector<Real> >      std::shared_ptrV;
  typedef SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a,b};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+2), bman );
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreateSimulatedVector( const std::shared_ptr<const Vector<Real> > &a,
                                                         const std::shared_ptr<const Vector<Real> > &b,
                                                         const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<const Vector<Real> >      std::shared_ptrV;
  typedef const SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a,b};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+2), bman );
}

template<class Real>
std::shared_ptr<Vector<Real> > CreateSimulatedVector( const std::shared_ptr<Vector<Real> > &a,
                                                   const std::shared_ptr<Vector<Real> > &b,
                                                   const std::shared_ptr<Vector<Real> > &c,
                                                   const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<Vector<Real> >      std::shared_ptrV;
  typedef SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a,b,c};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+3), bman );
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreateSimulatedVector( const std::shared_ptr<const Vector<Real> > &a,
                                                         const std::shared_ptr<const Vector<Real> > &b,
                                                         const std::shared_ptr<const Vector<Real> > &c,
                                                         const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<const Vector<Real> >      std::shared_ptrV;
  typedef const SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a,b,c};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+3), bman );
}

template<class Real>
std::shared_ptr<Vector<Real> > CreateSimulatedVector( const std::shared_ptr<Vector<Real> > &a,
                                                   const std::shared_ptr<Vector<Real> > &b,
                                                   const std::shared_ptr<Vector<Real> > &c,
                                                   const std::shared_ptr<Vector<Real> > &d,
                                                   const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<Vector<Real> >      std::shared_ptrV;
  typedef SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a,b,c,d};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+4), bman );
}

template<class Real>
std::shared_ptr<const Vector<Real> > CreateSimulatedVector( const std::shared_ptr<const Vector<Real> > &a,
                                                         const std::shared_ptr<const Vector<Real> > &b,
                                                         const std::shared_ptr<const Vector<Real> > &c,
                                                         const std::shared_ptr<const Vector<Real> > &d,
                                                         const std::shared_ptr<BatchManager<Real> > &bman ) {
  
  
  typedef std::shared_ptr<const Vector<Real> >      std::shared_ptrV;
  typedef const SimulatedVector<Real> PV;

  std::shared_ptrV temp[] = {a,b,c,d};
  return std::make_shared<PV( std::vector<std::shared_ptrV>>(temp, temp+4), bman );
}

} // namespace ROL

#endif // ROL_SIMULATED_VECTOR_H

