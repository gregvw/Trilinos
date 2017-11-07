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

/*! \file  constraint.hpp
    \brief Defines the SimOpt constraint for the 'poisson' example.
*/

#ifndef ROL_PDEOPT_LINEARPDECONSTRAINT_H
#define ROL_PDEOPT_LINEARPDECONSTRAINT_H

#include "ROL_Constraint_SimOpt.hpp"
#include "pde.hpp"
#include "assembler.hpp"
#include "solver.hpp"
#include "pdevector.hpp"

// Do not instantiate the template in this translation unit.
extern template class Assembler<double>;

//// Global Timers.
#ifdef ROL_TIMERS
namespace ROL {
  namespace PDEOPT {
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintSolverConstruct_Jacobian1        = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Solver Construction Time for Jacobian1");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintSolverConstruct_AdjointJacobian1 = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Solver Construction Time for Adjoint Jacobian1");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintSolverSolve_Jacobian1            = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Solver Solution Time for Jacobian1");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintSolverSolve_AdjointJacobian1     = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Solver Solution Time for Adjoint Jacobian1");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintApplyJacobian1                   = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Apply Jacobian1");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintApplyJacobian2                   = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Apply Jacobian2");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintApplyJacobian3                   = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Apply Jacobian3");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintApplyAdjointJacobian1            = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Apply Adjoint Jacobian1");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintApplyAdjointJacobian2            = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Apply Adjoint Jacobian2");
    std::shared_ptr<Teuchos::Time> LinearPDEConstraintApplyAdjointJacobian3            = Teuchos::TimeMonitor::getNewCounter("ROL::PDEOPT: Linear PDE Constraint Apply Adjoint Jacobian3");
  }
}
#endif


template<class Real>
class Linear_PDE_Constraint : public ROL::Constraint_SimOpt<Real> {
private:
  const std::shared_ptr<PDE<Real> > pde_;
  std::shared_ptr<Assembler<Real> > assembler_;
  std::shared_ptr<Solver<Real> >    solver_;
  bool initZvec_, initZpar_, isStochastic_;
  bool assembleRHS_, assembleJ1_, assembleJ2_, assembleJ3_, setSolver_;

  std::shared_ptr<Tpetra::MultiVector<> > cvec_;
  std::shared_ptr<Tpetra::MultiVector<> > uvec_;
  std::shared_ptr<Tpetra::MultiVector<> > zvec_;
  std::shared_ptr<std::vector<Real> >     zpar_;

  std::shared_ptr<Tpetra::MultiVector<> > vecR_;
  std::shared_ptr<Tpetra::CrsMatrix<> >   matJ1_;
  std::shared_ptr<Tpetra::CrsMatrix<> >   matJ2_;
  std::shared_ptr<Tpetra::MultiVector<> > vecJ3_;

  void assemble(const std::shared_ptr<const Tpetra::MultiVector<> > &zf,
                const std::shared_ptr<const std::vector<Real> > &zp) {
    // Initialize field component of z.
    if (initZvec_ && zf != nullptr) {
      zvec_ = assembler_->createControlVector();
      zvec_->putScalar(static_cast<Real>(0));
    }
    initZvec_ = false;
    // Initialize parameter component of z.
    if (initZpar_ && zp != nullptr) {
      zpar_ = std::make_shared<std::vector<Real>(zp->size(),static_cast<Real>>(0));
    }
    initZpar_ = false;
    // Assemble affine term.
    if (assembleRHS_) {
      assembler_->assemblePDEResidual(vecR_,pde_,uvec_,zvec_,zpar_);
    }
    assembleRHS_ = false;
    // Assemble jacobian_1.
    if (assembleJ1_) {
      assembler_->assemblePDEJacobian1(matJ1_,pde_,uvec_,zvec_,zpar_);
    }
    assembleJ1_ = false;
    // Assemble jacobian_2.
    if (assembleJ2_ && zf != nullptr) {
      assembler_->assemblePDEJacobian2(matJ2_,pde_,uvec_,zvec_,zpar_);
    }
    assembleJ2_ = false;
    // Assemble jacobian_3.
    if (assembleJ3_ && zp != nullptr) {
      assembler_->assemblePDEJacobian3(vecJ3_,pde_,uvec_,zvec_,zpar_);
    }
    assembleJ3_ = false;
  }

  void setSolver(void) {
    solver_->setA(matJ1_);
    setSolver_ = false;
  }

  void solveForward(std::shared_ptr<Tpetra::MultiVector<> > &x,
                    const std::shared_ptr<const Tpetra::MultiVector<> > &b) {
    if (setSolver_) {
      #ifdef ROL_TIMERS
        Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintSolverConstruct_Jacobian1);
      #endif
      setSolver();
    }

    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintSolverSolve_Jacobian1);
    #endif
    solver_->solve(x,b,false);
  }

  void solveAdjoint(std::shared_ptr<Tpetra::MultiVector<> > &x,
                    const std::shared_ptr<const Tpetra::MultiVector<> > &b) {
    if (setSolver_) {
      #ifdef ROL_TIMERS
        Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintSolverConstruct_AdjointJacobian1);
      #endif
      setSolver();
    }

    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintSolverSolve_AdjointJacobian1);
    #endif
    solver_->solve(x,b,true);
  }

  void applyJacobian1(const std::shared_ptr<Tpetra::MultiVector<> > &Jv,
                      const std::shared_ptr<const Tpetra::MultiVector<> > &v) {
    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintApplyJacobian1);
    #endif
    matJ1_->apply(*v,*Jv);
  }

  void applyAdjointJacobian1(const std::shared_ptr<Tpetra::MultiVector<> > &Jv,
                             const std::shared_ptr<const Tpetra::MultiVector<> > &v) {
    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintApplyAdjointJacobian1);
    #endif
    matJ1_->apply(*v,*Jv,Teuchos::TRANS);
  }

  void applyJacobian2(const std::shared_ptr<Tpetra::MultiVector<> > &Jv,
                      const std::shared_ptr<const Tpetra::MultiVector<> > &v) {
    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintApplyJacobian2);
    #endif
    matJ2_->apply(*v,*Jv);
  }

  void applyAdjointJacobian2(const std::shared_ptr<Tpetra::MultiVector<> > &Jv,
                             const std::shared_ptr<const Tpetra::MultiVector<> > &v) {
    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintApplyAdjointJacobian2);
    #endif
    matJ2_->apply(*v,*Jv,Teuchos::TRANS);
  }

  // Application routines
  void applyJacobian3(const std::shared_ptr<Tpetra::MultiVector<> > &Jv,
                      const std::shared_ptr<const std::vector<Real> > &v,
                      const bool zeroOut = true) const {
    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintApplyJacobian3);
    #endif
    if ( zeroOut ) {
      Jv->putScalar(static_cast<Real>(0));
    }
    const size_t size = static_cast<size_t>(v->size());
    for (size_t i = 0; i < size; ++i) {
      Teuchos::ArrayView<const size_t> col(&i,1);
      Jv->update((*v)[i],*(vecJ3_->subView(col)),static_cast<Real>(1));
    }
  }

  void applyAdjointJacobian3(const std::shared_ptr<std::vector<Real> > &Jv,
                             const std::shared_ptr<const Tpetra::MultiVector<> > &v) const {
    #ifdef ROL_TIMERS
      Teuchos::TimeMonitor LocalTimer(*ROL::PDEOPT::LinearPDEConstraintApplyAdjointJacobian3);
    #endif
    Teuchos::Array<Real> val(1,0);
    const size_t size = static_cast<size_t>(Jv->size());
    for (size_t i = 0; i < size; ++i) {
      Teuchos::ArrayView<const size_t> col(&i,1);
      vecJ3_->subView(col)->dot(*v, val.view(0,1));
      (*Jv)[i] = val[0];
    }
  }

public:
  virtual ~Linear_PDE_Constraint() {}

  Linear_PDE_Constraint(const std::shared_ptr<PDE<Real> > &pde,
                        const std::shared_ptr<MeshManager<Real> > &meshMgr,
                        const std::shared_ptr<const Teuchos::Comm<int> > &comm,
                        Teuchos::ParameterList &parlist,
                        std::ostream &outStream = std::cout,
                        const bool isStochastic = false)
    : ROL::Constraint_SimOpt<Real>(), pde_(pde),
      initZvec_(true), initZpar_(true), isStochastic_(isStochastic),
      assembleRHS_(true), assembleJ1_(true), assembleJ2_(true), assembleJ3_(true), setSolver_(true) {
    // Construct assembler.
    assembler_ = std::make_shared<Assembler<Real>(pde_->getFields>(),meshMgr,comm,parlist,outStream);
    assembler_->setCellNodes(*pde_);
    // Construct solver.
    solver_ = std::make_shared<Solver<Real>(parlist.sublist>("Solver"));
    // Initialize zero vectors.
    cvec_ = assembler_->createResidualVector();
    uvec_ = assembler_->createStateVector();
    uvec_->putScalar(static_cast<Real>(0));
  }

  void setParameter(const std::vector<Real> &param) {
    if (isStochastic_) {
      ROL::Constraint_SimOpt<Real>::setParameter(param);
      pde_->setParameter(param);
      assembleRHS_ = true;
      assembleJ1_  = true;
      assembleJ2_  = true;
      assembleJ3_  = true;
      setSolver_   = true;
    }
  }

  const std::shared_ptr<Assembler<Real> > getAssembler(void) const {
    return assembler_;
  }

  const std::shared_ptr<PDE<Real> > getPDE(void) const {
    return pde_;
  }

  using ROL::Constraint_SimOpt<Real>::update_1;
  void update_1(const ROL::Vector<Real> &u, bool flag = true, int iter = -1) {}

  using ROL::Constraint_SimOpt<Real>::update_2;
  void update_2(const ROL::Vector<Real> &z, bool flag = true, int iter = -1) {}

  using ROL::Constraint_SimOpt<Real>::update;
  void update(const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, bool flag = true, int iter = -1) {
    update_1(u,flag,iter);
    update_2(z,flag,iter);
  }

  using ROL::Constraint_SimOpt<Real>::value;
  void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >       cf = getField(c);
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);

    const Real one(1);
    cf->scale(one,*vecR_);

    matJ1_->apply(*uf,*cvec_);
    cf->update(one,*cvec_,one);
    if (zf != nullptr) {
      matJ2_->apply(*zf,*cvec_);
      cf->update(one,*cvec_,one);
    }
    if (zp != nullptr) {
      applyJacobian3(cvec_,zp,false);
      cf->update(one,*cvec_,one);
    }
  }

  void solve(ROL::Vector<Real> &c, ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >       cf = getField(c);
    std::shared_ptr<Tpetra::MultiVector<> >       uf = getField(u);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);

    const Real one(1);
    cf->scale(one,*vecR_);
    if (zf != nullptr) {
      matJ2_->apply(*zf,*cvec_);
      cf->update(one,*cvec_,one);
    }
    if (zp != nullptr) {
      applyJacobian3(cvec_,zp,false);
      cf->update(one,*cvec_,one);
    }

    solveForward(uf,cf);
    uf->scale(-one);

    matJ1_->apply(*uf,*cvec_);
    cf->update(one,*cvec_,one);
  }

  void applyJacobian_1(ROL::Vector<Real> &jv,
                 const ROL::Vector<Real> &v,
                 const ROL::Vector<Real> &u,
                 const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >      jvf = getField(jv);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);
    applyJacobian1(jvf,vf);
  }


  void applyJacobian_2(ROL::Vector<Real> &jv,
                 const ROL::Vector<Real> &v,
                 const ROL::Vector<Real> &u,
                 const ROL::Vector<Real> &z, Real &tol) {
    jv.zero();
    std::shared_ptr<Tpetra::MultiVector<> >      jvf = getField(jv);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);

    const Real one(1);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    if (vf != nullptr) {
      applyJacobian2(cvec_,vf);
      jvf->update(one,*cvec_,one);
    }
    std::shared_ptr<const std::vector<Real> >     vp = getConstParameter(v);
    bool zeroOut = (vf == nullptr);
    if (vp != nullptr) {
      applyJacobian3(cvec_,vp,zeroOut);
      jvf->update(one,*cvec_,one);
    }
  }


  void applyAdjointJacobian_1(ROL::Vector<Real> &ajv,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >     ajvf = getField(ajv);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);
    applyAdjointJacobian1(ajvf,vf);
  }


  void applyAdjointJacobian_2(ROL::Vector<Real> &ajv,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >     ajvf = getField(ajv);
    std::shared_ptr<std::vector<Real> >         ajvp = getParameter(ajv);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);

    if (zf != nullptr) {
      applyAdjointJacobian2(ajvf,vf);
    }
    if (zp != nullptr) {
      applyAdjointJacobian3(ajvp,vf);
    }
  }


  void applyInverseJacobian_1(ROL::Vector<Real> &ijv,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >     ijvf = getField(ijv);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);
    solveForward(ijvf,vf);
  }


  void applyInverseAdjointJacobian_1(ROL::Vector<Real> &iajv,
                               const ROL::Vector<Real> &v,
                               const ROL::Vector<Real> &u,
                               const ROL::Vector<Real> &z, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >    iajvf = getField(iajv);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<const Tpetra::MultiVector<> > zf = getConstField(z);
    std::shared_ptr<const std::vector<Real> >     zp = getConstParameter(z);

    assemble(zf,zp);
    solveAdjoint(iajvf,vf);
  }


  void applyAdjointHessian_11(ROL::Vector<Real> &ahwv,
                        const ROL::Vector<Real> &w,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }


  void applyAdjointHessian_12(ROL::Vector<Real> &ahwv,
                        const ROL::Vector<Real> &w,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }


  void applyAdjointHessian_21(ROL::Vector<Real> &ahwv,
                        const ROL::Vector<Real> &w,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }


  void applyAdjointHessian_22(ROL::Vector<Real> &ahwv,
                        const ROL::Vector<Real> &w,
                        const ROL::Vector<Real> &v,
                        const ROL::Vector<Real> &u,
                        const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }

  /***************************************************************************/
  /* Output routines.                                                        */
  /***************************************************************************/
  void printMeshData(std::ostream &outStream) const {
    assembler_->printMeshData(outStream);
  }

  void outputTpetraData() const {
    Tpetra::MatrixMarket::Writer< Tpetra::CrsMatrix<> > matWriter;
    if (matJ1_ != nullptr) {
      matWriter.writeSparseFile("jacobian1.txt", matJ1_);
    }
    else {
      std::ofstream emptyfile;
      emptyfile.open("jacobian1.txt");
      emptyfile.close();
    }
    if (matJ2_ != nullptr) {
      matWriter.writeSparseFile("jacobian2.txt", matJ2_);
    }
    else {
      std::ofstream emptyfile;
      emptyfile.open("jacobian2.txt");
      emptyfile.close();
    }
    if (vecR_ != nullptr) {
      matWriter.writeDenseFile("residual.txt", vecR_);
    }
    else {
      std::ofstream emptyfile;
      emptyfile.open("residual.txt");
      emptyfile.close();
    }
  }

  void outputTpetraVector(const std::shared_ptr<const Tpetra::MultiVector<> > &vec,
                          const std::string &filename) const {
    Tpetra::MatrixMarket::Writer< Tpetra::CrsMatrix<> > vecWriter;
    vecWriter.writeDenseFile(filename, vec);
  }
  /***************************************************************************/
  /* End of output routines.                                                 */
  /***************************************************************************/

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
