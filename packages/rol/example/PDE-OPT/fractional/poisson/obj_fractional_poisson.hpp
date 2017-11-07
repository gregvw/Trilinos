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

/*! \file  obj.hpp
    \brief Provides the interface for local (cell-based) objective function computations.
*/

#ifndef PDEOPT_QOI_FRACTIONAL_POISSON_HPP
#define PDEOPT_QOI_FRACTIONAL_POISSON_HPP

#include "../../TOOLS/qoi.hpp"
#include "pde_fractional_poisson.hpp"

template <class Real>
class QoI_L2Tracking_Fractional_Poisson : public QoI<Real> {
private:
  std::shared_ptr<FE<Real> > fe_;

  std::shared_ptr<Intrepid::FieldContainer<Real> > target_;

  Real targetFunc(const std::vector<Real> & x) const {
    int size = x.size();
    Real val(0);
    for (int i = 0; i < size; ++i) {
      val += x[i]*x[i];
    }
    return val;
  }

public:
  QoI_L2Tracking_Fractional_Poisson(const std::shared_ptr<FE<Real> > &fe) : fe_(fe) {
    int c = fe_->cubPts()->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    int d = fe_->cubPts()->dimension(2);
    std::vector<Real> pt(d);
    target_ = std::make_shared<Intrepid::FieldContainer<Real>>(c,p);
    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < p; ++j) {
        for (int k = 0; k < d; ++k) {
          pt[k] = (*fe_->cubPts())(i,j,k);
        }
        (*target_)(i,j) = targetFunc(pt);
      }
    }
  }

  Real value(std::shared_ptr<Intrepid::FieldContainer<Real> > & val,
             const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
             const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
             const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // Get relevant dimensions
    int c = u_coeff->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    // Initialize output val
    val = std::make_shared<Intrepid::FieldContainer<Real>>(c);
    // Evaluate state on FE basis
    std::shared_ptr<Intrepid::FieldContainer<Real> > valU_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_->evaluateValue(valU_eval, u_coeff);
    // Compute difference between state and target
    Intrepid::RealSpaceTools<Real>::subtract(*valU_eval,*target_);
    // Compute squared L2-norm of diff
    fe_->computeIntegral(val,valU_eval,valU_eval);
    // Scale by one half
    Intrepid::RealSpaceTools<Real>::scale(*val,static_cast<Real>(0.5));
    return static_cast<Real>(0);
  }

  void gradient_1(std::shared_ptr<Intrepid::FieldContainer<Real> > & grad,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // Get relevant dimensions
    int c = u_coeff->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    int f = fe_->N()->dimension(1);
    // Initialize output grad
    grad = std::make_shared<Intrepid::FieldContainer<Real>>(c, f);
    // Evaluate state on FE basis
    std::shared_ptr<Intrepid::FieldContainer<Real> > valU_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_->evaluateValue(valU_eval, u_coeff);
    // Compute difference between state and target
    Intrepid::RealSpaceTools<Real>::subtract(*valU_eval,*target_);
    // Compute gradient of squared L2-norm of diff
    Intrepid::FunctionSpaceTools::integrate<Real>(*grad,
                                                  *valU_eval,
                                                  *(fe_->NdetJ()),
                                                  Intrepid::COMP_CPP, false);
  }

  void gradient_2(std::shared_ptr<Intrepid::FieldContainer<Real> > & grad,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Tracking_Poisson::gradient_2 is zero.");
  }

  void HessVec_11(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    int c = v_coeff->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    int f = fe_->N()->dimension(1);
    std::shared_ptr<Intrepid::FieldContainer<Real> > valV_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    hess = std::make_shared<Intrepid::FieldContainer<Real>>(c, f);
    fe_->evaluateValue(valV_eval, v_coeff);
    Intrepid::FunctionSpaceTools::integrate<Real>(*hess,
                                                  *valV_eval,
                                                  *(fe_->NdetJ()),
                                                  Intrepid::COMP_CPP, false);
  }

  void HessVec_12(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Tracking_Poisson::HessVec_12 is zero.");
  }

  void HessVec_21(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Tracking_Poisson::HessVec_21 is zero.");
  }

  void HessVec_22(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Tracking_Poisson::HessVec_22 is zero.");
  }

}; // QoI_L2Tracking

template <class Real>
class QoI_L2Penalty_Fractional_Poisson : public QoI<Real> {
private:
  std::shared_ptr<FE<Real> > fe_;

public:
  QoI_L2Penalty_Fractional_Poisson(const std::shared_ptr<FE<Real> > &fe) : fe_(fe) {}

  Real value(std::shared_ptr<Intrepid::FieldContainer<Real> > & val,
             const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
             const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
             const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // Get relevant dimensions
    int c = z_coeff->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    // Initialize output val
    val = std::make_shared<Intrepid::FieldContainer<Real>>(c);
    // Build local state tracking term
    std::shared_ptr<Intrepid::FieldContainer<Real> > valZ_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_->evaluateValue(valZ_eval, z_coeff);
    fe_->computeIntegral(val,valZ_eval,valZ_eval);
    Intrepid::RealSpaceTools<Real>::scale(*val,static_cast<Real>(0.5));
    return static_cast<Real>(0);
  }

  void gradient_1(std::shared_ptr<Intrepid::FieldContainer<Real> > & grad,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Tracking_Poisson::gradient_1 is zero.");
  }

  void gradient_2(std::shared_ptr<Intrepid::FieldContainer<Real> > & grad,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // Get relevant dimensions
    int c = z_coeff->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    int f = fe_->N()->dimension(1);
    // Initialize output grad
    grad = std::make_shared<Intrepid::FieldContainer<Real>>(c, f);
    // Build local gradient of state tracking term
    std::shared_ptr<Intrepid::FieldContainer<Real> > valZ_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_->evaluateValue(valZ_eval, z_coeff);
    Intrepid::FunctionSpaceTools::integrate<Real>(*grad,
                                                  *valZ_eval,
                                                  *(fe_->NdetJ()),
                                                  Intrepid::COMP_CPP, false);
  }

  void HessVec_11(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Penalty_Poisson::HessVec_11 is zero.");
  }

  void HessVec_12(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Penalty_Poisson::HessVec_12 is zero.");
  }

  void HessVec_21(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> QoI_L2Penalty_Poisson::HessVec_21 is zero.");
  }

  void HessVec_22(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & v_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    int c = v_coeff->dimension(0);
    int p = fe_->cubPts()->dimension(1);
    int f = fe_->N()->dimension(1);
    std::shared_ptr<Intrepid::FieldContainer<Real> > valV_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    hess = std::make_shared<Intrepid::FieldContainer<Real>>(c, f);
    fe_->evaluateValue(valV_eval, v_coeff);
    Intrepid::FunctionSpaceTools::integrate<Real>(*hess,
                                                  *valV_eval,
                                                  *(fe_->NdetJ()),
                                                  Intrepid::COMP_CPP, false);
  }

}; // QoI_L2Penalty

#endif
