
#ifndef ROL_PDEOPT_ENERGY_OBJECTIVE
#define ROL_PDEOPT_ENERGY_OBJECTIVE

#include "ROL_Objective.hpp"
#include "../TOOLS/assembler.hpp"

// Do not instantiate the template in this translation unit.
extern template class Assembler<double>;

template<class Real>
class EnergyObjective : public ROL::Objective<Real> {
private:
  const std::shared_ptr<PDE<Real> > pde_;
  std::shared_ptr<Assembler<Real> > assembler_;
  bool assembleRHS_, assembleJ1_;

  std::shared_ptr<Tpetra::MultiVector<> > cvec_;
  std::shared_ptr<Tpetra::MultiVector<> > uvec_;

  std::shared_ptr<Tpetra::MultiVector<> > res_;
  std::shared_ptr<Tpetra::CrsMatrix<> >   jac_;

  void assemble(void) {
    // Assemble affine term.
    if (assembleRHS_) {
      assembler_->assemblePDEResidual(res_,pde_,uvec_);
    }
    assembleRHS_ = false;
    // Assemble jacobian_1.
    if (assembleJ1_) {
      assembler_->assemblePDEJacobian1(jac_,pde_,uvec_);
    }
    assembleJ1_ = false;
  }

public:
  EnergyObjective(const std::shared_ptr<PDE<Real> > &pde,
                  const std::shared_ptr<MeshManager<Real> > &meshMgr,
                  const std::shared_ptr<const Teuchos::Comm<int> > &comm,
                  Teuchos::ParameterList &parlist,
                  std::ostream &outStream = std::cout)
    : pde_(pde), assembleRHS_(true), assembleJ1_(true) {
    // Construct assembler.
    assembler_ = std::make_shared<Assembler<Real>(pde_->getFields>(),meshMgr,comm,parlist,outStream);
    assembler_->setCellNodes(*pde_);
    // Initialize zero vectors.
    cvec_ = assembler_->createResidualVector();
    uvec_ = assembler_->createStateVector();
    uvec_->putScalar(static_cast<Real>(0));
    assemble();
  }

  const std::shared_ptr<Assembler<Real> > getAssembler(void) const {
    return assembler_;
  }

  Real value(const ROL::Vector<Real> &u, Real &tol) {
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    const Real half(0.5), one(1);
    jac_->apply(*uf,*cvec_);
    cvec_->update(one,*res_,half);
    Teuchos::Array<Real> val(1,0);
    cvec_->dot(*uf, val.view(0,1));
    return val[0];
  }

  void gradient(ROL::Vector<Real> &g, const ROL::Vector<Real> &u, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >       gf = getField(g);
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    const Real one(1);
    gf->scale(one,*res_);
    jac_->apply(*uf,*cvec_);
    gf->update(one,*cvec_,one);
  }

  void hessVec(ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u, Real &tol) {
    std::shared_ptr<Tpetra::MultiVector<> >      hvf = getField(hv);
    std::shared_ptr<const Tpetra::MultiVector<> > vf = getConstField(v);
    std::shared_ptr<const Tpetra::MultiVector<> > uf = getConstField(u);
    jac_->apply(*vf,*hvf);
  }

  void precond(ROL::Vector<Real> &Pv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u, Real &tol) {
    Pv.set(v.dual());
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

};

#endif
