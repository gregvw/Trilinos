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

#ifndef ROL_FRACTIONAL_OBJECTIVE_SIMOPT_H
#define ROL_FRACTIONAL_OBJECTIVE_SIMOPT_H

#include "ROL_Objective_SimOpt.hpp"

template <class Real>
class FractionalObjective : public ROL::Objective_SimOpt<Real> {
private:
  const std::shared_ptr<ROL::Objective_SimOpt<Real> > obj_;

public:
  FractionalObjective(const std::shared_ptr<ROL::Objective_SimOpt<Real> > &obj)
    : obj_(obj) {}

  void setParameter(const std::vector<Real> &param) {
    ROL::Objective_SimOpt<Real>::setParameter(param);
    obj_->setParameter(param);
  }

  void update( const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, bool flag = true, int iter = -1 ) {
    obj_->update(u,z,flag,iter);
  }

  Real value( const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    return obj_->value(*ur,z,tol);
  }

  void gradient_1( ROL::Vector<Real> &g, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<Tpetra::MultiVector<> > gf = getField(g);
    std::shared_ptr<ROL::Vector<Real> > gr
      = std::make_shared<ROL::TpetraMultiVector<Real>(gf->subViewNonConst(cols>()));
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    g.zero();
    obj_->gradient_1(*gr,*ur,z,tol);
    //std::shared_ptr<Tpetra::MultiVector<Real> > grf = getField(*gr);
    //gf->getVectorNonConst(0)->scale(static_cast<Real>(1),*grf);
  }

  void gradient_2( ROL::Vector<Real> &g, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    obj_->gradient_2(g,*ur,z,tol);
  }

  void hessVec_11( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<Tpetra::MultiVector<> > hvf = getField(hv);
    std::shared_ptr<ROL::Vector<Real> > hvr
      = std::make_shared<ROL::TpetraMultiVector<Real>(hvf->subViewNonConst(cols>()));
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<ROL::Vector<Real> > vr
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(vf)->subViewNonConst(cols())));
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    obj_->hessVec_11(*hvr,*vr,*ur,z,tol);
  }

  void hessVec_12( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<Tpetra::MultiVector<> > hvf = getField(hv);
    std::shared_ptr<ROL::Vector<Real> > hvr
      = std::make_shared<ROL::TpetraMultiVector<Real>(hvf->subViewNonConst(cols>()));
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    obj_->hessVec_12(*hvr,v,*ur,z,tol);
  }

  void hessVec_21( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
                   const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<ROL::Vector<Real> > vr
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(vf)->subViewNonConst(cols())));
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    obj_->hessVec_21(hv,*vr,*ur,z,tol);
  }

  void hessVec_22( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, 
             const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    Teuchos::Array<size_t> cols(1,0);
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<ROL::Vector<Real> > ur
      = Teuchos::rcp(new ROL::TpetraMultiVector<Real>(
          std::const_pointer_cast<Tpetra::MultiVector<> >(uf)->subViewNonConst(cols())));
    obj_->hessVec_22(hv,v,*ur,z,tol);
  }

private: // Vector accessor functions

  std::shared_ptr<const Tpetra::MultiVector<> > getConstField(const ROL::Vector<Real> &x) const {
    return dynamic_cast<const ROL::TpetraMultiVector<Real>&>(x).getVector();
  }

  std::shared_ptr<Tpetra::MultiVector<> > getField(ROL::Vector<Real> &x) const {
    return dynamic_cast<ROL::TpetraMultiVector<Real>&>(x).getVector();
  }

}; // class Objective_SimOpt

#endif
