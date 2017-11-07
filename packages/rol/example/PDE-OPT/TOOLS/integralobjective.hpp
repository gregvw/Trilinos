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

#ifndef PDE_INTEGRALOBJECTIVE_HPP
#define PDE_INTEGRALOBJECTIVE_HPP

#include "ROL_Objective_SimOpt.hpp"
#include "qoi.hpp"
#include "assembler.hpp"
#include "pdevector.hpp"

// Do not instantiate the template in this translation unit.
extern template class Assembler<double>;

template<class Real>
class IntegralObjective : public ROL::Objective_SimOpt<Real> {
private:
  const std::shared_ptr<QoI<Real> > qoi_;
  const std::shared_ptr<Assembler<Real> > assembler_;

  std::shared_ptr<Tpetra::MultiVector<> > vecG1_;
  std::shared_ptr<Tpetra::MultiVector<> > vecG2_;
  std::shared_ptr<std::vector<Real> >     vecG3_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH11_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH12_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH13_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH21_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH22_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH23_;
  std::shared_ptr<std::vector<Real> >     vecH31_;
  std::shared_ptr<std::vector<Real> >     vecH32_;
  std::shared_ptr<std::vector<Real> >     vecH33_;

public:
  IntegralObjective(const std::shared_ptr<QoI<Real> > &qoi,
                    const std::shared_ptr<Assembler<Real> > &assembler)
    : qoi_(qoi), assembler_(assembler) {}

  void setParameter(const std::vector<Real> &param) {
    ROL::Objective_SimOpt<Real>::setParameter(param);
    qoi_->setParameter(param);
  }

  Real value(const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
    return assembler_->assembleQoIValue(qoi_,uf,zf,zp);
  }

  void gradient_1(ROL::Vector<Real> &g, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol ) {
    g.zero();
    try {
      std::shared_ptr<Tpetra::MultiVector<> >       gf = getField(g);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIGradient1(vecG1_,qoi_,uf,zf,zp);
      gf->scale(static_cast<Real>(1),*vecG1_);
    }
    catch ( Exception::Zero & ez ) {
      g.zero();
    }
    catch ( Exception::NotImplemented & eni ) {
      ROL::Objective_SimOpt<Real>::gradient_1(g,u,z,tol);
    }
  }

  void gradient_2(ROL::Vector<Real> &g, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol ) {
    int NotImplemented(0), IsZero(0);
    g.zero();
    // Compute control field gradient
    try {
      // Get state and control vectors
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      std::shared_ptr<Tpetra::MultiVector<> >       gf = getField(g);
      assembler_->assembleQoIGradient2(vecG2_,qoi_,uf,zf,zp);
      gf->scale(static_cast<Real>(1),*vecG2_);
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Compute control parameter gradient
    try {
      // Get state and control vectors
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      std::shared_ptr<std::vector<Real> >           gp = getParameter(g);
      assembler_->assembleQoIGradient3(vecG3_,qoi_,uf,zf,zp);
      gp->assign(vecG3_->begin(),vecG3_->end());
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Zero gradient
    if ( IsZero == 2 || (IsZero == 1 && NotImplemented == 1) ) {
      g.zero();
    }
    // Not Implemented
    if ( NotImplemented == 2 ) {
      ROL::Objective_SimOpt<Real>::gradient_2(g,u,z,tol);
    }
  }

  void hessVec_11( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    hv.zero();
    try {
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec11(vecH11_,qoi_,vf,uf,zf,zp);
      hvf->scale(static_cast<Real>(1),*vecH11_);
    }
    catch (Exception::Zero &ez) {
      hv.zero();
    }
    catch (Exception::NotImplemented &eni) {
      ROL::Objective_SimOpt<Real>::hessVec_11(hv,v,u,z,tol);
    }
  }

  void hessVec_12( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
                   const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    int NotImplemented(0), IsZero(0);
    hv.zero();
    // Compute state field/control field hessvec
    try {
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec12(vecH12_,qoi_,vf,uf,zf,zp);
      hvf->scale(static_cast<Real>(1),*vecH12_);
    }
    catch (Exception::Zero &ez) {
      hv.zero();
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      hv.zero();
      NotImplemented++;
    }
    // Compute state field/control parameter hessvec
    try {
      // Get state and control vectors
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const std::vector<Real> >     vp = getConstParameter(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec13(vecH13_,qoi_,vp,uf,zf,zp);
      hvf->update(static_cast<Real>(1),*vecH13_,static_cast<Real>(1));
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Zero hessvec
    if ( IsZero == 2 || (IsZero == 1 && NotImplemented == 1) ) {
      hv.zero();
    }
    // Not Implemented
    if ( NotImplemented == 2 ) {
      ROL::Objective_SimOpt<Real>::hessVec_12(hv,v,u,z,tol);
    }
  }

  void hessVec_21( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
                   const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    int NotImplemented(0), IsZero(0);
    hv.zero();
    // Compute control field/state field hessvec
    try {
      // Get state and control vectors
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec21(vecH21_,qoi_,vf,uf,zf,zp);
      hvf->scale(static_cast<Real>(1),*vecH21_);
    }
    catch ( Exception::Zero & ez ) {
      hv.zero();
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      hv.zero();
      NotImplemented++;
    }
    // Compute control parameter/state field hessvec
    try {
      // Get state and control vectors
      std::shared_ptr<std::vector<Real> >          hvp = getParameter(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec31(vecH31_,qoi_,vf,uf,zf,zp);
      hvp->assign(vecH31_->begin(),vecH31_->end());
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Zero hessvec
    if ( IsZero == 2 || (IsZero == 1 && NotImplemented == 1) ) {
      hv.zero();
    }
    // Not Implemented
    if ( NotImplemented == 2 ) {
      ROL::Objective_SimOpt<Real>::hessVec_21(hv,v,u,z,tol);
    }
  }

  void hessVec_22( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    int NotImplemented(0), IsZero(0);
    hv.zero();
    // Compute control field/field hessvec
    try {
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec22(vecH22_,qoi_,vf,uf,zf,zp);
      hvf->scale(static_cast<Real>(1),*vecH22_);
    }
    catch (Exception::Zero &ez) {
      hv.zero();
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      hv.zero();
      NotImplemented++;
    }
    // Compute control field/parameter hessvec
    try {
      // Get state and control vectors
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const std::vector<Real> >     vp = getConstParameter(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec23(vecH23_,qoi_,vp,uf,zf,zp);
      hvf->update(static_cast<Real>(1),*vecH23_,static_cast<Real>(1));
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Compute control parameter/field hessvec
    try {
      std::shared_ptr<std::vector<Real> >          hvp = getParameter(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec32(vecH32_,qoi_,vf,uf,zf,zp);
      hvp->assign(vecH32_->begin(),vecH32_->end());
    }
    catch (Exception::Zero &ez) {
      std::shared_ptr<std::vector<Real> > hvp = getParameter(hv);
      if ( hvp != nullptr ) {
        const int size = hvp->size();
        hvp->assign(size,static_cast<Real>(0));
      }
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      std::shared_ptr<std::vector<Real> > hvp = getParameter(hv);
      if ( hvp != nullptr ) {
        const int size = hvp->size();
        hvp->assign(size,static_cast<Real>(0));
      }
      NotImplemented++;
    }
    // Compute control parameter/parameter hessvec
    try {
      std::shared_ptr<std::vector<Real> >          hvp = getParameter(hv);
      std::shared_ptr<const std::vector<Real> >     vp = getConstParameter(v);
      std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec33(vecH33_,qoi_,vp,uf,zf,zp);
      const int size = hvp->size();
      for (int i = 0; i < size; ++i) {
        (*hvp)[i] += (*vecH33_)[i];
      }
    }
    catch (Exception::Zero &ez) {
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      NotImplemented++;
    }

    // Zero hessvec
    if ( IsZero > 0 && (IsZero + NotImplemented == 4) ) {
      hv.zero();
    }
    // Not Implemented
    if ( NotImplemented == 4 ) {
      ROL::Objective_SimOpt<Real>::hessVec_22(hv,v,u,z,tol);
    }
  }

private: // Vector accessor functions

  std::shared_ptr<const Tpetra::MultiVector<> > getConstField(const ROL::Vector<Real> &x) const {
    std::shared_ptr<const Tpetra::MultiVector<> > xp;
    try {
      xp = dynamic_cast<const ROL::TpetraMultiVector<Real>&>(x).getVector();
    }
    catch (std::exception &e) {
      std::shared_ptr<const ROL::TpetraMultiVector<Real> > xvec
        = dynamic_cast<const PDE_OptVector<Real>&>(x).getField();
      if (xvec == nullptr) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    return xp;
  }

  std::shared_ptr<Tpetra::MultiVector<> > getField(ROL::Vector<Real> &x) const {
    std::shared_ptr<Tpetra::MultiVector<> > xp;
    try {
      xp = dynamic_cast<ROL::TpetraMultiVector<Real>&>(x).getVector();
    }
    catch (std::exception &e) {
      std::shared_ptr<ROL::TpetraMultiVector<Real> > xvec
        = dynamic_cast<PDE_OptVector<Real>&>(x).getField();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    return xp;
  }

  std::shared_ptr<const std::vector<Real> > getConstParameter(const ROL::Vector<Real> &x) const {
    std::shared_ptr<const std::vector<Real> > xp;
    try {
      std::shared_ptr<const ROL::StdVector<Real> > xvec
        = dynamic_cast<const PDE_OptVector<Real>&>(x).getParameter();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    catch (std::exception &e) {
      xp = nullptr;
    }
    return xp;
  }

  std::shared_ptr<std::vector<Real> > getParameter(ROL::Vector<Real> &x) const {
    std::shared_ptr<std::vector<Real> > xp;
    try {
      std::shared_ptr<ROL::StdVector<Real> > xvec
        = dynamic_cast<PDE_OptVector<Real>&>(x).getParameter();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    catch (std::exception &e) {
      xp = nullptr;
    }
    return xp;
  }
};


template<class Real>
class IntegralOptObjective : public ROL::Objective<Real> {
private:
  const std::shared_ptr<QoI<Real> > qoi_;
  const std::shared_ptr<Assembler<Real> > assembler_;

  std::shared_ptr<Tpetra::MultiVector<> > vecG2_;
  std::shared_ptr<std::vector<Real> >     vecG3_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH22_;
  std::shared_ptr<Tpetra::MultiVector<> > vecH23_;
  std::shared_ptr<std::vector<Real> >     vecH32_;
  std::shared_ptr<std::vector<Real> >     vecH33_;

public:
  IntegralOptObjective(const std::shared_ptr<QoI<Real> > &qoi,
                       const std::shared_ptr<Assembler<Real> > &assembler)
    : qoi_(qoi), assembler_(assembler) {}

  void setParameter(const std::vector<Real> &param) {
    ROL::Objective<Real>::setParameter(param);
    qoi_->setParameter(param);
  }

  Real value(const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
    return assembler_->assembleQoIValue(qoi_,nullptr,zf,zp);
  }

  void gradient(ROL::Vector<Real> &g,
                const ROL::Vector<Real> &z, Real &tol ) {
    int NotImplemented(0), IsZero(0);
    g.zero();
    // Compute control field gradient
    try {
      // Get state and control vectors
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      std::shared_ptr<Tpetra::MultiVector<> >       gf = getField(g);
      assembler_->assembleQoIGradient2(vecG2_,qoi_,nullptr,zf,zp);
      gf->scale(static_cast<Real>(1),*vecG2_);
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Compute control parameter gradient
    try {
      // Get state and control vectors
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      std::shared_ptr<std::vector<Real> >           gp = getParameter(g);
      assembler_->assembleQoIGradient3(vecG3_,qoi_,nullptr,zf,zp);
      gp->assign(vecG3_->begin(),vecG3_->end());
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Zero gradient
    if ( IsZero == 2 || (IsZero == 1 && NotImplemented == 1) ) {
      g.zero();
    }
    // Not Implemented
    if ( NotImplemented == 2 ) {
      ROL::Objective<Real>::gradient(g,z,tol);
    }
  }

  void hessVec(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
               const ROL::Vector<Real> &z, Real &tol ) {
    int NotImplemented(0), IsZero(0);
    hv.zero();
    // Compute control field/field hessvec
    try {
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec22(vecH22_,qoi_,vf,nullptr,zf,zp);
      hvf->scale(static_cast<Real>(1),*vecH22_);
    }
    catch (Exception::Zero &ez) {
      hv.zero();
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      hv.zero();
      NotImplemented++;
    }
    // Compute control field/parameter hessvec
    try {
      // Get state and control vectors
      std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
      std::shared_ptr<const std::vector<Real> >     vp = getConstParameter(v);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec23(vecH23_,qoi_,vp,nullptr,zf,zp);
      hvf->update(static_cast<Real>(1),*vecH23_,static_cast<Real>(1));
    }
    catch ( Exception::Zero & ez ) {
      IsZero++;
    }
    catch ( Exception::NotImplemented & eni ) {
      NotImplemented++;
    }
    // Compute control parameter/field hessvec
    try {
      std::shared_ptr<std::vector<Real> >          hvp = getParameter(hv);
      std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec32(vecH32_,qoi_,vf,nullptr,zf,zp);
      hvp->assign(vecH32_->begin(),vecH32_->end());
    }
    catch (Exception::Zero &ez) {
      std::shared_ptr<std::vector<Real> > hvp = getParameter(hv);
      if ( hvp != nullptr ) {
        const int size = hvp->size();
        hvp->assign(size,static_cast<Real>(0));
      }
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      std::shared_ptr<std::vector<Real> > hvp = getParameter(hv);
      if ( hvp != nullptr ) {
        const int size = hvp->size();
        hvp->assign(size,static_cast<Real>(0));
      }
      NotImplemented++;
    }
    // Compute control parameter/parameter hessvec
    try {
      std::shared_ptr<std::vector<Real> >          hvp = getParameter(hv);
      std::shared_ptr<const std::vector<Real> >     vp = getConstParameter(v);
      std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
      std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);
      assembler_->assembleQoIHessVec33(vecH33_,qoi_,vp,nullptr,zf,zp);
      const int size = hvp->size();
      for (int i = 0; i < size; ++i) {
        (*hvp)[i] += (*vecH33_)[i];
      }
    }
    catch (Exception::Zero &ez) {
      IsZero++;
    }
    catch (Exception::NotImplemented &eni) {
      NotImplemented++;
    }

    // Zero hessvec
    if ( IsZero > 0 && (IsZero + NotImplemented == 4) ) {
      hv.zero();
    }
    // Not Implemented
    if ( NotImplemented == 4 ) {
      ROL::Objective<Real>::hessVec(hv,v,z,tol);
    }
  }

private: // Vector accessor functions

  std::shared_ptr<const Tpetra::MultiVector<> > getConstField(const ROL::Vector<Real> &x) const {
    std::shared_ptr<const Tpetra::MultiVector<> > xp;
    try {
      xp = dynamic_cast<const ROL::TpetraMultiVector<Real>&>(x).getVector();
    }
    catch (std::exception &e) {
      std::shared_ptr<const ROL::TpetraMultiVector<Real> > xvec
        = dynamic_cast<const PDE_OptVector<Real>&>(x).getField();
      if (xvec == nullptr) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    return xp;
  }

  std::shared_ptr<Tpetra::MultiVector<> > getField(ROL::Vector<Real> &x) const {
    std::shared_ptr<Tpetra::MultiVector<> > xp;
    try {
      xp = dynamic_cast<ROL::TpetraMultiVector<Real>&>(x).getVector();
    }
    catch (std::exception &e) {
      std::shared_ptr<ROL::TpetraMultiVector<Real> > xvec
        = dynamic_cast<PDE_OptVector<Real>&>(x).getField();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    return xp;
  }

  std::shared_ptr<const std::vector<Real> > getConstParameter(const ROL::Vector<Real> &x) const {
    std::shared_ptr<const std::vector<Real> > xp;
    try {
      std::shared_ptr<const ROL::StdVector<Real> > xvec
        = dynamic_cast<const PDE_OptVector<Real>&>(x).getParameter();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    catch (std::exception &e) {
      xp = nullptr;
    }
    return xp;
  }

  std::shared_ptr<std::vector<Real> > getParameter(ROL::Vector<Real> &x) const {
    std::shared_ptr<std::vector<Real> > xp;
    try {
      std::shared_ptr<ROL::StdVector<Real> > xvec
        = dynamic_cast<PDE_OptVector<Real>&>(x).getParameter();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    catch (std::exception &e) {
      xp = nullptr;
    }
    return xp;
  }
};

template<class Real>
class IntegralAffineSimObjective : public ROL::Objective_SimOpt<Real> {
private:
  const std::shared_ptr<QoI<Real> > qoi_;
  const std::shared_ptr<Assembler<Real> > assembler_;

  Real shift_;
  std::shared_ptr<Tpetra::MultiVector<> > vecG1_;
  std::shared_ptr<Tpetra::MultiVector<> > ufield_;
  std::shared_ptr<Tpetra::MultiVector<> > zfield_;
  std::shared_ptr<std::vector<Real> >     zparam_;

  bool isAssembled_;

  void assemble(const std::shared_ptr<const Tpetra::MultiVector<> > &zf,
                const std::shared_ptr<const std::vector<Real> >     &zp) {
    if ( !isAssembled_ ) {
      ufield_ = assembler_->createStateVector();
      ufield_->putScalar(static_cast<Real>(0));
      if ( zf != nullptr ) {
        zfield_ = assembler_->createControlVector();
        zfield_->putScalar(static_cast<Real>(0));
      }
      if ( zp != nullptr ) {
       zparam_ = std::make_shared<std::vector<Real>(zp->size(),static_cast<Real>>(0));
      }
      shift_ = assembler_->assembleQoIValue(qoi_,ufield_,zfield_,zparam_);
      assembler_->assembleQoIGradient1(vecG1_,qoi_,ufield_,zfield_,zparam_);
      isAssembled_ = true;
    }
  }

public:
  IntegralAffineSimObjective(const std::shared_ptr<QoI<Real> > &qoi,
                             const std::shared_ptr<Assembler<Real> > &assembler)
    : qoi_(qoi), assembler_(assembler), isAssembled_(false) {}

  void setParameter(const std::vector<Real> &param) {
    ROL::Objective_SimOpt<Real>::setParameter(param);
    qoi_->setParameter(param);
    isAssembled_ = false;
  }

  Real value(const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);

    const size_t n = uf->getNumVectors();
    Teuchos::Array<Real> val(n,0);
    vecG1_->dot(*uf, val.view(0,n));
    
    return val[0] + shift_;
  }

  void gradient_1(ROL::Vector<Real> &g, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol ) {
    std::shared_ptr<Tpetra::MultiVector<> >       gf = getField(g);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);

    gf->scale(static_cast<Real>(1),*vecG1_);
  }

  void gradient_2(ROL::Vector<Real> &g, const ROL::Vector<Real> &u,
                  const ROL::Vector<Real> &z, Real &tol ) {
    g.zero();
  }

  void hessVec_11( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    hv.zero();
  }

  void hessVec_12( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
                   const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    hv.zero();
  }

  void hessVec_21( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
                   const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    hv.zero();
  }

  void hessVec_22( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    hv.zero();
  }

private: // Vector accessor functions

  std::shared_ptr<const Tpetra::MultiVector<> > getConstField(const ROL::Vector<Real> &x) const {
    std::shared_ptr<const Tpetra::MultiVector<> > xp;
    try {
      xp = dynamic_cast<const ROL::TpetraMultiVector<Real>&>(x).getVector();
    }
    catch (std::exception &e) {
      std::shared_ptr<const ROL::TpetraMultiVector<Real> > xvec
        = dynamic_cast<const PDE_OptVector<Real>&>(x).getField();
      if (xvec == nullptr) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    return xp;
  }

  std::shared_ptr<Tpetra::MultiVector<> > getField(ROL::Vector<Real> &x) const {
    std::shared_ptr<Tpetra::MultiVector<> > xp;
    try {
      xp = dynamic_cast<ROL::TpetraMultiVector<Real>&>(x).getVector();
    }
    catch (std::exception &e) {
      std::shared_ptr<ROL::TpetraMultiVector<Real> > xvec
        = dynamic_cast<PDE_OptVector<Real>&>(x).getField();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    return xp;
  }

  std::shared_ptr<const std::vector<Real> > getConstParameter(const ROL::Vector<Real> &x) const {
    std::shared_ptr<const std::vector<Real> > xp;
    try {
      std::shared_ptr<const ROL::StdVector<Real> > xvec
        = dynamic_cast<const PDE_OptVector<Real>&>(x).getParameter();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    catch (std::exception &e) {
      xp = nullptr;
    }
    return xp;
  }

  std::shared_ptr<std::vector<Real> > getParameter(ROL::Vector<Real> &x) const {
    std::shared_ptr<std::vector<Real> > xp;
    try {
      std::shared_ptr<ROL::StdVector<Real> > xvec
        = dynamic_cast<PDE_OptVector<Real>&>(x).getParameter();
      if ( xvec == nullptr ) {
        xp = nullptr;
      }
      else {
        xp = xvec->getVector();
      }
    }
    catch (std::exception &e) {
      xp = nullptr;
    }
    return xp;
  }
};

#endif
