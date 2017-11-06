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

/** \file
    \brief  Contains definitions for W. Hock and K. Schittkowski 2nd test function.
    \author Created by D. Ridzal and D. Kouri.
 */

#ifndef USE_HESSVEC 
#define USE_HESSVEC 1
#endif

#ifndef ROL_HS2_HPP
#define ROL_HS2_HPP

#include "ROL_StdVector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_Types.hpp"

namespace ROL {
namespace ZOO {

  /** \brief W. Hock and K. Schittkowski 2nd test function.
   */
  template<class Real>
  class Objective_HS2 : public Objective<Real> {
 
    typedef std::vector<Real> vector;
    typedef Vector<Real>      V;
    typedef StdVector<Real>   SV;

  private:
    
    std::shared_ptr<const vector> getVector( const V& x ) {
      
      return dynamic_cast<const SV>(x).getVector(); 
    }

    std::shared_ptr<vector> getVector( V& x ) {
      
      return dynamic_cast<SV>(x).getVector();
    }

  public:
    Objective_HS2(void) {}

    Real value( const Vector<Real> &x, Real &tol ) {

      
      std::shared_ptr<const vector> ex = getVector(x); 
      return static_cast<Real>(100) * std::pow((*ex)[1] - std::pow((*ex)[0],2),2)
           + std::pow(static_cast<Real>(1)-(*ex)[0],2);
    }

    void gradient( Vector<Real> &g, const Vector<Real> &x, Real &tol ) {

      
      std::shared_ptr<const vector> ex = getVector(x);
      std::shared_ptr<vector> eg = getVector(g);
      (*eg)[0] = static_cast<Real>(-400) * ((*ex)[1] - std::pow((*ex)[0],2))
               * (*ex)[0] - static_cast<Real>(2) * (static_cast<Real>(1)-(*ex)[0]);
      (*eg)[1] = static_cast<Real>(200) * ((*ex)[1] - std::pow((*ex)[0],2)); 
    }
#if USE_HESSVEC
    void hessVec( Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &x, Real &tol ) {

      
      std::shared_ptr<const vector> ex = getVector(x);
      std::shared_ptr<const vector> ev = getVector(v);
      std::shared_ptr<vector> ehv = getVector(hv);
 
      Real h11 = static_cast<Real>(-400) * (*ex)[1]
               + static_cast<Real>(1200) * std::pow((*ex)[0],2)
               + static_cast<Real>(2); 
      Real h22 = static_cast<Real>(200);
      Real h12 = static_cast<Real>(-400) * (*ex)[0];
      Real h21 = static_cast<Real>(-400) * (*ex)[0];

      Real alpha(0);

      (*ehv)[0] = (h11+alpha) * (*ev)[0] + h12 * (*ev)[1];
      (*ehv)[1] = h21 * (*ev)[0] + (h22+alpha) * (*ev)[1];
    } 
#endif
    void invHessVec( Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &x, Real &tol ) {
 
      
      std::shared_ptr<const vector> ex = getVector(x);
      std::shared_ptr<const vector> ev = getVector(v);
      std::shared_ptr< vector> ehv = getVector(hv);
     
      Real h11 = static_cast<Real>(-400) * (*ex)[1]
               + static_cast<Real>(1200) * std::pow((*ex)[0],2)
               + static_cast<Real>(2); 
      Real h22 = static_cast<Real>(200);
      Real h12 = static_cast<Real>(-400) * (*ex)[0];
      Real h21 = static_cast<Real>(-400) * (*ex)[0];
  
      (*ehv)[0] = static_cast<Real>(1)/(h11*h22 - h12*h21)
                * (h22 * (*ev)[0] - h12 * (*ev)[1]);
      (*ehv)[1] = static_cast<Real>(1)/(h11*h22 - h12*h21)
                * (-h21 * (*ev)[0] + h11 * (*ev)[1]);
    }
  };

template<class Real>
void getHS2( std::shared_ptr<Objective<Real> >       &obj,
             std::shared_ptr<BoundConstraint<Real> > &con, 
             std::shared_ptr<Vector<Real> >          &x0,
             std::shared_ptr<Vector<Real> >          &x ) {
  // Problem dimension
  int n = 2;

  // Get Initial Guess
  std::shared_ptr<std::vector<Real> > x0p = std::make_shared<std::vector<Real>>(n,0.0);
  (*x0p)[0] = -2.0; (*x0p)[1] = 1.0;
  x0 = std::make_shared<StdVector<Real>>(x0p);

  // Get Solution
  std::shared_ptr<std::vector<Real> > xp = std::make_shared<std::vector<Real>>(n,0.0);
  Real a = std::sqrt(598.0/1200.0);
  Real b = 400.0 * std::pow(a,3.0);
  (*xp)[0] = 2.0*a*std::cos(1.0/3.0 * std::acos(1.0/b));
  (*xp)[1] = 1.5;
  x = std::make_shared<StdVector<Real>>(xp);

  // Instantiate Objective Function
  obj = std::make_shared<Objective_HS2<Real>>();

  // Instantiate BoundConstraint
  std::shared_ptr<std::vector<Real> > lp = std::make_shared<std::vector<Real>>(n,0.0);
  (*lp)[0] = ROL_NINF<Real>(); (*lp)[1] = 1.5;
  std::shared_ptr<Vector<Real> > l = std::make_shared<StdVector<Real>>(lp);
  std::shared_ptr<std::vector<Real> > up = std::make_shared<std::vector<Real>>(n,0.0);
  (*up)[0] = ROL_INF<Real>(); (*up)[1] = ROL_INF<Real>(); 
  std::shared_ptr<Vector<Real> > u = std::make_shared<StdVector<Real>>(up);
  con = std::make_shared<Bounds<Real>>(l,u);
}

} // End ZOO Namespace
} // End ROL Namespace

#endif
