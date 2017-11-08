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

#ifndef ROL_ScalarMinimizationLineSearch_H
#define ROL_ScalarMinimizationLineSearch_H

/** \class ROL::ScalarMinimizationLineSearch
    \brief Implements line search methods that attempt to minimize the
           scalar function \f$\phi(t) := f(x+ts)\f$.
*/

#include "ROL_LineSearch.hpp"
#include "ROL_BrentsScalarMinimization.hpp"
#include "ROL_BisectionScalarMinimization.hpp"
#include "ROL_GoldenSectionScalarMinimization.hpp"
#include "ROL_ScalarFunction.hpp"
#include "ROL_Bracketing.hpp"

namespace ROL {

template<class Real>
class ScalarMinimizationLineSearch : public LineSearch<Real> {
private:
  ROL::SharedPointer<Vector<Real> >             xnew_;
  ROL::SharedPointer<Vector<Real> >             g_;
  ROL::SharedPointer<ScalarMinimization<Real> > sm_;
  ROL::SharedPointer<Bracketing<Real> >         br_;
  ROL::SharedPointer<ScalarFunction<Real> >     sf_;

  ECurvatureCondition econd_;
  Real c1_;
  Real c2_;
  Real c3_;
  int max_nfval_;

  class Phi : public ScalarFunction<Real> {
  private:
    const ROL::SharedPointer<Vector<Real> > xnew_;
    const ROL::SharedPointer<Vector<Real> > g_;
    const ROL::SharedPointer<const Vector<Real> > x_;
    const ROL::SharedPointer<const Vector<Real> > s_;
    const ROL::SharedPointer<Objective<Real> > obj_;
    const ROL::SharedPointer<BoundConstraint<Real> > con_;
    Real ftol_;
    void updateIterate(Real alpha) {
      xnew_->set(*x_);
      xnew_->axpy(alpha,*s_);
      if ( con_->isActivated() ) {
        con_->project(*xnew_);
      }
    }
  public:
    Phi(const ROL::SharedPointer<Vector<Real> > &xnew,
        const ROL::SharedPointer<Vector<Real> > &g,
        const ROL::SharedPointer<const Vector<Real> > &x,
        const ROL::SharedPointer<const Vector<Real> > &s,
        const ROL::SharedPointer<Objective<Real> > &obj,
        const ROL::SharedPointer<BoundConstraint<Real> > &con)
     : xnew_(xnew), g_(g), x_(x), s_(s), obj_(obj), con_(con),
       ftol_(std::sqrt(ROL_EPSILON<Real>())) {}
    Real value(const Real alpha) {
      updateIterate(alpha);
      obj_->update(*xnew_);
      return obj_->value(*xnew_,ftol_);
    }
    Real deriv(const Real alpha) {
      updateIterate(alpha);
      obj_->update(*xnew_);
      obj_->gradient(*g_,*xnew_,ftol_);
      return s_->dot(g_->dual());
    }
  };

  class LineSearchStatusTest : public ScalarMinimizationStatusTest<Real> {
  private:
    ROL::SharedPointer<ScalarFunction<Real> > phi_;

    const Real f0_;
    const Real g0_;

    const Real c1_;
    const Real c2_;
    const Real c3_;
    const int max_nfval_;
    const ECurvatureCondition econd_;


  public:
    LineSearchStatusTest(const Real f0, const Real g0,
                         const Real c1, const Real c2, const Real c3,
                         const int max_nfval, ECurvatureCondition econd,
                         const ROL::SharedPointer<ScalarFunction<Real> > &phi)
      : phi_(phi), f0_(f0), g0_(g0), c1_(c1), c2_(c2), c3_(c3),
        max_nfval_(max_nfval), econd_(econd) {}

    bool check(Real &x, Real &fx, Real &gx,
               int &nfval, int &ngval, const bool deriv = false) {
      Real one(1), two(2);
      bool armijo = (fx <= f0_ + c1_*x*g0_);
//      bool itcond = (nfval >= max_nfval_);
      bool curvcond = false;
//      if (armijo && !itcond) {
      if (armijo) {
        if (econd_ == CURVATURECONDITION_GOLDSTEIN) {
          curvcond = (fx >= f0_ + (one-c1_)*x*g0_);
        }
        else if (econd_ == CURVATURECONDITION_NULL) {
          curvcond = true;
        }
        else {
          if (!deriv) {
            gx = phi_->deriv(x); ngval++;
          }
          if (econd_ == CURVATURECONDITION_WOLFE) {
            curvcond = (gx >= c2_*g0_);
          }
          else if (econd_ == CURVATURECONDITION_STRONGWOLFE) {
            curvcond = (std::abs(gx) <= c2_*std::abs(g0_));
          }
          else if (econd_ == CURVATURECONDITION_GENERALIZEDWOLFE) {
            curvcond = (c2_*g0_ <= gx && gx <= -c3_*g0_);
          }
          else if (econd_ == CURVATURECONDITION_APPROXIMATEWOLFE) {
            curvcond = (c2_*g0_ <= gx && gx <= (two*c1_ - one)*g0_);
          }
        }
      }
      //return (armijo && curvcond) || itcond;
      return (armijo && curvcond);
    }
  };

public:
  // Constructor
  ScalarMinimizationLineSearch( Teuchos::ParameterList &parlist,
    const ROL::SharedPointer<ScalarMinimization<Real> > &sm = ROL::nullPointer,
    const ROL::SharedPointer<Bracketing<Real> > &br = ROL::nullPointer,
    const ROL::SharedPointer<ScalarFunction<Real> > &sf  = ROL::nullPointer )
    : LineSearch<Real>(parlist) {
    Real zero(0), p4(0.4), p6(0.6), p9(0.9), oem4(1.e-4), oem10(1.e-10), one(1);
    Teuchos::ParameterList &list0 = parlist.sublist("Step").sublist("Line Search");
    Teuchos::ParameterList &list  = list0.sublist("Line-Search Method");
    // Get Bracketing Method
    if( br == ROL::nullPointer ) {
      br_ = ROL::makeShared<Bracketing<Real>>();
    }
    else {
      br_ = br;
    }
    // Get ScalarMinimization Method
    std::string type = list.get("Type","Brent's");
    Real tol         = list.sublist(type).get("Tolerance",oem10);
    int niter        = list.sublist(type).get("Iteration Limit",1000);
    Teuchos::ParameterList plist;
    plist.sublist("Scalar Minimization").set("Type",type);
    plist.sublist("Scalar Minimization").sublist(type).set("Tolerance",tol);
    plist.sublist("Scalar Minimization").sublist(type).set("Iteration Limit",niter);

    if( sm == ROL::nullPointer ) { // No user-provided ScalarMinimization object

      if ( type == "Brent's" ) {
        sm_ = ROL::makeShared<BrentsScalarMinimization<Real>>(plist);
      }
      else if ( type == "Bisection" ) {
        sm_ = ROL::makeShared<BisectionScalarMinimization<Real>>(plist);
      }
      else if ( type == "Golden Section" ) {
        sm_ = ROL::makeShared<GoldenSectionScalarMinimization<Real>>(plist);
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
          ">>> (ROL::ScalarMinimizationLineSearch): Undefined ScalarMinimization type!");
      }
    }
    else {
      sm_ = sm;
    }

    sf_ = sf;


    // Status test for line search
    econd_ = StringToECurvatureCondition(list0.sublist("Curvature Condition").get("Type","Strong Wolfe Conditions"));
    max_nfval_ = list0.get("Function Evaluation Limit",20);
    c1_        = list0.get("Sufficient Decrease Tolerance",oem4);
    c2_        = list0.sublist("Curvature Condition").get("General Parameter",p9);
    c3_        = list0.sublist("Curvature Condition").get("Generalized Wolfe Parameter",p6);
    // Check status test inputs
    c1_ = ((c1_ < zero) ? oem4 : c1_);
    c2_ = ((c2_ < zero) ? p9   : c2_);
    c3_ = ((c3_ < zero) ? p9   : c3_);
    if ( c2_ <= c1_ ) {
      c1_ = oem4;
      c2_ = p9;
    }
    EDescent edesc = StringToEDescent(list0.sublist("Descent Method").get("Type","Quasi-Newton Method"));
    if ( edesc == DESCENT_NONLINEARCG ) {
      c2_ = p4;
      c3_ = std::min(one-c2_,c3_);
    }
  }

  void initialize( const Vector<Real> &x, const Vector<Real> &s, const Vector<Real> &g,
                   Objective<Real> &obj, BoundConstraint<Real> &con ) {
    LineSearch<Real>::initialize(x,s,g,obj,con);
    xnew_ = x.clone();
    g_    = g.clone();
  }

  // Find the minimum of phi(alpha) = f(x + alpha*s) using Brent's method
  void run( Real &alpha, Real &fval, int &ls_neval, int &ls_ngrad,
            const Real &gs, const Vector<Real> &s, const Vector<Real> &x,
            Objective<Real> &obj, BoundConstraint<Real> &con ) {
    ls_neval = 0; ls_ngrad = 0;

    // Get initial line search parameter
    alpha = LineSearch<Real>::getInitialAlpha(ls_neval,ls_ngrad,fval,gs,x,s,obj,con);

    // Build ScalarFunction and ScalarMinimizationStatusTest
    ROL::SharedPointer<const Vector<Real> > x_ptr(&x);
    ROL::SharedPointer<const Vector<Real> > s_ptr(&s);
    ROL::SharedPointer<Objective<Real> > obj_ptr(&obj);
    ROL::SharedPointer<BoundConstraint<Real> > bnd_ptr(&con);


    ROL::SharedPointer<ScalarFunction<Real> > phi;

    if( sf_ == ROL::nullPointer ) {
      phi = ROL::makeShared<Phi>(xnew_,g_,x_ptr,s_ptr,obj_ptr,bnd_ptr);
    }
    else {
      phi = sf_;
    }

    ROL::SharedPointer<ScalarMinimizationStatusTest<Real> > test
      = ROL::makeShared<LineSearchStatusTest>(fval,gs,c1_,c2_,c3_,max_nfval_,econd_,phi);

    // Run Bracketing
    int nfval = 0, ngrad = 0;
    Real A(0),      fA = fval;
    Real B = alpha, fB = phi->value(B);
    br_->run(alpha,fval,A,fA,B,fB,nfval,ngrad,*phi,*test);
    B = alpha;
    ls_neval += nfval; ls_ngrad += ngrad;

    // Run ScalarMinimization
    nfval = 0, ngrad = 0;
    sm_->run(fval, alpha, nfval, ngrad, *phi, A, B, *test);
    ls_neval += nfval; ls_ngrad += ngrad;

    LineSearch<Real>::setNextInitialAlpha(alpha);
  }
};

}

#endif
