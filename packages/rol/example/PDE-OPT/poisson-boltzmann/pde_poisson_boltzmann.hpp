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

/*! \file  pde.hpp
    \brief Implements the local PDE interface for the Poisson-Boltzmann control problem.
*/

#ifndef PDE_POISSON_BOLTZMANN_HPP
#define PDE_POISSON_BOLTZMANN_HPP

#include "../TOOLS/pde.hpp"
#include "../TOOLS/fe.hpp"

#include "Intrepid_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_CellTools.hpp"

#include <memory>

template <class Real>
class PDE_Poisson_Boltzmann : public PDE<Real> {
private:
  // Finite element basis information
  std::shared_ptr<Intrepid::Basis<Real, Intrepid::FieldContainer<Real> > > basisPtr_;
  std::vector<std::shared_ptr<Intrepid::Basis<Real, Intrepid::FieldContainer<Real> > > > basisPtrs_;
  // Cell cubature information
  std::shared_ptr<Intrepid::Cubature<Real> > cellCub_;
  // Cell node information
  std::shared_ptr<Intrepid::FieldContainer<Real> > volCellNodes_;
  std::vector<std::vector<std::shared_ptr<Intrepid::FieldContainer<Real> > > > bdryCellNodes_;
  std::vector<std::vector<std::vector<int> > > bdryCellLocIds_;
  // Finite element definition
  std::shared_ptr<FE<Real> > fe_vol_;

public:
  PDE_Poisson_Boltzmann(Teuchos::ParameterList &parlist) {
    // Finite element fields.
    int basisOrder = parlist.sublist("Problem").get("Order of FE discretization",1);
    if (basisOrder == 1) {
      basisPtr_ = std::make_shared<Intrepid::Basis_HGRAD_QUAD_C1_FEM<Real, Intrepid::FieldContainer<Real> >>();
    }
    else if (basisOrder == 2) {
      basisPtr_ = std::make_shared<Intrepid::Basis_HGRAD_QUAD_C2_FEM<Real, Intrepid::FieldContainer<Real> >>();
    }
    basisPtrs_.clear(); basisPtrs_.push_back(basisPtr_);
    // Quadrature rules.
    shards::CellTopology cellType = basisPtr_->getBaseCellTopology();                  // get the cell type from any basis
    Intrepid::DefaultCubatureFactory<Real> cubFactory;                                 // create cubature factory
    int cubDegree = parlist.sublist("PDE Poisson Boltzmann").get("Cubature Degree",2); // set cubature degree, e.g., 2
    cellCub_ = cubFactory.create(cellType, cubDegree);                                 // create default cubature
  }

  void residual(std::shared_ptr<Intrepid::FieldContainer<Real> > & res,
                const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // GET DIMENSIONS
    int c = fe_vol_->N()->dimension(0);
    int f = fe_vol_->N()->dimension(1);
    int p = fe_vol_->N()->dimension(2);
    int d = cellCub_->getDimension();
    // INITIALIZE RESIDUAL
    res = std::make_shared<Intrepid::FieldContainer<Real>>(c, f);
    // COMPUTE STIFFNESS TERM
    std::shared_ptr<Intrepid::FieldContainer<Real> > gradU_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p, d);
    fe_vol_->evaluateGradient(gradU_eval, u_coeff);
    Intrepid::FunctionSpaceTools::integrate<Real>(*res,
                                                  *gradU_eval,
                                                  *(fe_vol_->gradNdetJ()),
                                                  Intrepid::COMP_CPP, false);
    // ADD NONLINEAR TERM
    std::shared_ptr<Intrepid::FieldContainer<Real> > valU_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_vol_->evaluateValue(valU_eval, u_coeff);
    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < p; ++j) {
        (*valU_eval)(i,j) = std::exp(-(*valU_eval)(i,j));
      }
    }
    Intrepid::FunctionSpaceTools::integrate<Real>(*res,
                                                  *valU_eval,
                                                  *(fe_vol_->NdetJ()),
                                                  Intrepid::COMP_CPP, true);
    // ADD CONTROL TERM
    if ( z_coeff != nullptr ) {
      std::shared_ptr<Intrepid::FieldContainer<Real> > valZ_eval =
        std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
      fe_vol_->evaluateValue(valZ_eval, z_coeff);
      Intrepid::RealSpaceTools<Real>::scale(*valZ_eval,static_cast<Real>(-1));
      Intrepid::FunctionSpaceTools::integrate<Real>(*res,
                                                    *valZ_eval,
                                                    *(fe_vol_->NdetJ()),
                                                    Intrepid::COMP_CPP, true);
    }
  }

  void Jacobian_1(std::shared_ptr<Intrepid::FieldContainer<Real> > & jac,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // GET DIMENSIONS
    int c = fe_vol_->N()->dimension(0);
    int f = fe_vol_->N()->dimension(1);
    int p = fe_vol_->N()->dimension(2);
    // INITIALIZE JACOBIAN
    jac = std::make_shared<Intrepid::FieldContainer<Real>>(c, f, f);
    // COMPUTE STIFFNESS TERM
    Intrepid::FunctionSpaceTools::integrate<Real>(*jac,
                                                  *(fe_vol_->gradN()),
                                                  *(fe_vol_->gradNdetJ()),
                                                  Intrepid::COMP_CPP, false);
    // ADD NONLINEAR TERM
    std::shared_ptr<Intrepid::FieldContainer<Real> > valU_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_vol_->evaluateValue(valU_eval, u_coeff);
    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < p; ++j) {
        (*valU_eval)(i,j) = -std::exp(-(*valU_eval)(i,j));
      }
    }
    Intrepid::FieldContainer<Real> NexpU(c,f,p);
    Intrepid::FunctionSpaceTools::scalarMultiplyDataField<Real>(NexpU,
                                                                *valU_eval,
                                                                *(fe_vol_->N()));
    Intrepid::FunctionSpaceTools::integrate<Real>(*jac,
                                                  NexpU,
                                                  *(fe_vol_->NdetJ()),
                                                  Intrepid::COMP_CPP, true);
  }

  void Jacobian_2(std::shared_ptr<Intrepid::FieldContainer<Real> > & jac,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    if ( z_coeff != nullptr ) {
      // GET DIMENSIONS
      int c = fe_vol_->N()->dimension(0);
      int f = fe_vol_->N()->dimension(1);
      // INITIALIZE JACOBIAN
      jac = std::make_shared<Intrepid::FieldContainer<Real>>(c, f, f);
      // ADD CONTROL TERM
      Intrepid::FunctionSpaceTools::integrate<Real>(*jac,
                                                    *(fe_vol_->N()),
                                                    *(fe_vol_->NdetJ()),
                                                    Intrepid::COMP_CPP, false);
      Intrepid::RealSpaceTools<Real>::scale(*jac,static_cast<Real>(-1));
    }
  }

  void Hessian_11(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & l_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    // GET DIMENSIONS
    int c = fe_vol_->N()->dimension(0);
    int f = fe_vol_->N()->dimension(1);
    int p = fe_vol_->N()->dimension(2);
    // INITIALIZE HESSIAN
    hess = std::make_shared<Intrepid::FieldContainer<Real>>(c, f, f);
    // COMPUTE NONLINEAR TERM
    std::shared_ptr<Intrepid::FieldContainer<Real> > valU_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_vol_->evaluateValue(valU_eval, u_coeff);
    std::shared_ptr<Intrepid::FieldContainer<Real> > valL_eval =
      std::make_shared<Intrepid::FieldContainer<Real>>(c, p);
    fe_vol_->evaluateValue(valL_eval, l_coeff);
    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < p; ++j) {
        (*valU_eval)(i,j) = (*valL_eval)(i,j)*std::exp(-(*valU_eval)(i,j));
      }
    }
    Intrepid::FieldContainer<Real> NLexpU(c,f,p);
    Intrepid::FunctionSpaceTools::scalarMultiplyDataField<Real>(NLexpU,
                                                                *valU_eval,
                                                                *(fe_vol_->N()));
    Intrepid::FunctionSpaceTools::integrate<Real>(*hess,
                                                  NLexpU,
                                                  *(fe_vol_->NdetJ()),
                                                  Intrepid::COMP_CPP, false);
  }

  void Hessian_12(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & l_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> (PDE_Poisson_Boltzmann:Hessian_12: Hessian is zero.");
  }

  void Hessian_21(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & l_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> (PDE_Poisson_Boltzmann:Hessian_21: Hessian is zero.");
  }

  void Hessian_22(std::shared_ptr<Intrepid::FieldContainer<Real> > & hess,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & l_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & u_coeff,
                  const std::shared_ptr<const Intrepid::FieldContainer<Real> > & z_coeff = nullptr,
                  const std::shared_ptr<const std::vector<Real> > & z_param = nullptr) {
    throw Exception::Zero(">>> (PDE_Poisson_Boltzmann:Hessian_22: Hessian is zero.");
  }

  void RieszMap_1(std::shared_ptr<Intrepid::FieldContainer<Real> > & riesz) {
    // GET DIMENSIONS
    int c = fe_vol_->N()->dimension(0);
    int f = fe_vol_->N()->dimension(1);
    // INITIALIZE RIESZ
    riesz = std::make_shared<Intrepid::FieldContainer<Real>>(c, f, f);
    *riesz = *fe_vol_->stiffMat();
    Intrepid::RealSpaceTools<Real>::add(*riesz,*(fe_vol_->massMat()));
  }

  void RieszMap_2(std::shared_ptr<Intrepid::FieldContainer<Real> > & riesz) {
    // GET DIMENSIONS
    int c = fe_vol_->N()->dimension(0);
    int f = fe_vol_->N()->dimension(1);
    // INITIALIZE RIESZ
    riesz = std::make_shared<Intrepid::FieldContainer<Real>>(c, f, f);
    *riesz = *fe_vol_->massMat();
  }

  std::vector<std::shared_ptr<Intrepid::Basis<Real, Intrepid::FieldContainer<Real> > > > getFields() {
    return basisPtrs_;
  }

  void setCellNodes(const std::shared_ptr<Intrepid::FieldContainer<Real> > &volCellNodes,
                    const std::vector<std::vector<std::shared_ptr<Intrepid::FieldContainer<Real> > > > &bdryCellNodes,
                    const std::vector<std::vector<std::vector<int> > > &bdryCellLocIds) {
    volCellNodes_ = volCellNodes;
    bdryCellNodes_ = bdryCellNodes;
    bdryCellLocIds_ = bdryCellLocIds;
    // Finite element definition.
    fe_vol_ = std::make_shared<FE<Real>>(volCellNodes_,basisPtr_,cellCub_);
  }

  const std::shared_ptr<FE<Real> > getFE(void) const {
    return fe_vol_;
  }

}; // PDE_Poisson_Boltzmann

#endif
