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

#ifndef ROL_BOUND_CONSTRAINT_PARTITIONED_H
#define ROL_BOUND_CONSTRAINT_PARTITIONED_H

#include "ROL_BoundConstraint.hpp"
#include "ROL_PartitionedVector.hpp"
#include "ROL_Types.hpp"
#include <iostream>

/** @ingroup func_group
    \class ROL::BoundConstraint_Partitioned
    \brief A composite composite BoundConstraint formed
           from bound constraints on subvectors of a PartitionedVector

*/
namespace ROL {

template <class Real>
class BoundConstraint_Partitioned : public BoundConstraint<Real> {

  typedef Vector<Real>                          V;
  typedef PartitionedVector<Real>               PV;
  typedef typename std::vector<Real>::size_type uint;

private:
  std::vector<std::shared_ptr<BoundConstraint<Real> > > bnd_;

  std::shared_ptr<V> l_;
  std::shared_ptr<V> u_;

  uint dim_;
 
public:
  ~BoundConstraint_Partitioned() {}

  BoundConstraint_Partitioned(const std::vector<std::shared_ptr<BoundConstraint<Real> > > &bnd)
    : bnd_(bnd), dim_(bnd.size()) {
    BoundConstraint<Real>::deactivate();
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        BoundConstraint<Real>::activate();
        break;
      }
    }
    std::vector<std::shared_ptr<Vector<Real> > > lp(dim_);
    std::vector<std::shared_ptr<Vector<Real> > > up(dim_);
    for( uint k=0; k<dim_; ++k ) {
      lp[k] = bnd[k]->getLowerBound()->clone();
      lp[k]->set(*bnd_[k]->getLowerBound());
      up[k] = bnd[k]->getUpperBound()->clone();
      up[k]->set(*bnd_[k]->getUpperBound());
    }
    l_ = std::make_shared<PV>(lp);
    u_ = std::make_shared<PV>(up);
  }

  void update( const Vector<Real> &x, bool flag = true, int iter = -1 ) {
    const PV &xpv = dynamic_cast<const PV>(x);
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->update(*(xpv.get(k)),flag,iter);   
      }
    }
  }

  void project( Vector<Real> &x ) {
    PV &xpv = dynamic_cast<PV>(x);
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->project(*xpv.get(k));
      }
    }
  }

  void projectInterior( Vector<Real> &x ) {
    PV &xpv = dynamic_cast<PV>(x);
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->projectInterior(*xpv.get(k));
      }
    }
  }

  void pruneUpperActive( Vector<Real> &v, const Vector<Real> &x, Real eps = 0.0 ) {
          PV &vpv = dynamic_cast<PV>(v);
    const PV &xpv = dynamic_cast<const PV>(x);
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->pruneUpperActive(*(vpv.get(k)),*(xpv.get(k)),eps);
      }
    }
 }

  void pruneUpperActive( Vector<Real> &v, const Vector<Real> &g, const Vector<Real> &x, Real eps = 0.0 ) {
          PV &vpv = dynamic_cast<PV>(v);
    const PV &gpv = dynamic_cast<const PV>(g);
    const PV &xpv = dynamic_cast<const PV>(x);
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->pruneUpperActive(*(vpv.get(k)),*(gpv.get(k)),*(xpv.get(k)),eps);
      }
    }
  }
 
  void pruneLowerActive( Vector<Real> &v, const Vector<Real> &x, Real eps = 0.0 ) {
          PV &vpv = dynamic_cast<PV>(v);
    const PV &xpv = dynamic_cast<const PV>(x);
   for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->pruneLowerActive(*(vpv.get(k)),*(xpv.get(k)),eps);
      }
    }
  }

  void pruneLowerActive( Vector<Real> &v, const Vector<Real> &g, const Vector<Real> &x, Real eps = 0.0 ) {
          PV &vpv = dynamic_cast<PV>(v);
    const PV &gpv = dynamic_cast<const PV>(g);
    const PV &xpv = dynamic_cast<const PV>(x);
    for( uint k=0; k<dim_; ++k ) {
      if( bnd_[k]->isActivated() ) {
        bnd_[k]->pruneLowerActive(*(vpv.get(k)),*(gpv.get(k)),*(xpv.get(k)),eps);
      }
    }
  }
 
  const std::shared_ptr<const Vector<Real> > getLowerBound( void ) const {
    return l_;
  }
       
  const std::shared_ptr<const Vector<Real> > getUpperBound( void ) const {
    return u_;
  }

  bool isFeasible( const Vector<Real> &v ) { 
    bool feasible = true;
    const PV &vs = dynamic_cast<const PV>(v);
    for( uint k=0; k<dim_; ++k ) {
      if(bnd_[k]->isActivated()) {
        feasible = feasible && bnd_[k]->isFeasible(*(vs.get(k)));
      }
    }
    return feasible;
  }
}; // class BoundConstraint_Partitioned



template<class Real>
std::shared_ptr<BoundConstraint<Real> > 
CreateBoundConstraint_Partitioned( const std::shared_ptr<BoundConstraint<Real> > &bnd1,
                                   const std::shared_ptr<BoundConstraint<Real> > &bnd2 ) {

     
  typedef BoundConstraint<Real>             BND;
  typedef BoundConstraint_Partitioned<Real> BNDP;
  std::shared_ptr<BND> temp[] = {bnd1, bnd2};
  return std::make_shared<BNDP( std::vector<std::shared_ptr<BND> >>(temp,temp+2) );
}


} // namespace ROL

#endif
