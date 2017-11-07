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

/*! \file  example_01.cpp
    \brief Solves a source inversion problem governed by the
           advection-diffusion equation.  Performs optimal
           experimental design (OED).
*/

#include "Teuchos_Comm.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Version.hpp"

#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_CompositeStep.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_TpetraMultiVector.hpp"

#include <iostream>
#include <algorithm>

#include "data.hpp"
#include "objective.hpp"
#include "constraint.hpp"
#include "ROL_ExperimentDesignObjective.hpp"

typedef double RealT;

class MyInterfaceOED : public ROL::ExperimentDesignInterface<RealT> {
public:
  MyInterfaceOED(const std::shared_ptr<ROL::Objective_SimOpt<RealT> > &obj,
                 const std::shared_ptr<ROL::Constraint_SimOpt<RealT> > &con,
                 const std::shared_ptr<ROL::Vector<RealT> > &state,
                 const std::shared_ptr<ROL::Vector<RealT> > &stateDual,
                 const std::shared_ptr<ROL::Vector<RealT> > &control,
                 const std::shared_ptr<ROL::Vector<RealT> > &controlDual,
                 const std::shared_ptr<ROL::Vector<RealT> > &constraint,
                 const std::shared_ptr<ROL::Vector<RealT> > &constraintDual,
                 const std::shared_ptr<ROL::Vector<RealT> > &observation,
                 const std::shared_ptr<ROL::Vector<RealT> > &observationDual,
                 const std::vector<std::shared_ptr<ROL::Vector<RealT> > > &randvecs,
                 const std::vector<std::shared_ptr<ROL::Vector<RealT> > > &training) :
    ExperimentDesignInterface<RealT>(obj, con, state, stateDual, control, controlDual, constraint, constraintDual, observation, observationDual, randvecs, training) {}

  // Override interface functions; in this case, they are the same.

  virtual void applyObserveOp(ROL::Vector<RealT> &obsv, const ROL::Vector<RealT> &v) const {
    obsv.set(v);
  }

  virtual void applyAdjointObserveOp(ROL::Vector<RealT> &aobsv, const ROL::Vector<RealT> &v) const {
    aobsv.set(v);
  }

  virtual void applyWeightOp(ROL::Vector<RealT> &weightv, const ROL::Vector<RealT> &v, const ROL::Vector<RealT> &w) const {
    weightv.set(v.dual());
    weightv.applyBinary(ROL::Elementwise::Multiply<RealT>(), w);
  }

}; // class MyInterfaceOED

// Used for elementwise is-greather-than?
template<class Real>
class IsGreaterThan : public ROL::Elementwise::UnaryFunction<Real> {
private:
  const Real num_, yes_, no_;
public:
  IsGreaterThan(const Real &num, const Real &yes, const Real &no) : num_(num), yes_(yes), no_(no) {}
  Real apply(const Real &x) const {
    if (x > num_) {
      return static_cast<Real>(yes_);
    }
    else {
      return static_cast<Real>(no_);
    }
  }
}; // class IsGreaterThan

int main(int argc, char *argv[]) {

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  /*** Initialize communicator. ***/
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &bhs);
  std::shared_ptr<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  const int myRank = comm->getRank();
  if ((iprint > 0) && (myRank == 0)) {
    outStream = &std::cout, false;
  }
  else {
    outStream = &bhs, false;
  }

  int errorFlag  = 0;

  // *** Example body.
  try {

    /*** Read in XML input ***/
    std::string filename    = "input.xml";
    std::string filenameOED = "inputOED.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    std::shared_ptr<Teuchos::ParameterList> parlistOED = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    Teuchos::updateParametersFromXmlFile( filenameOED, parlistOED.ptr() );

    /*** Initialize main data structure. ***/
    std::shared_ptr<PoissonData<RealT> > data = std::make_shared<PoissonData<RealT>>(comm, parlist, outStream);

    // Get random weights parameter.
    RealT fnzw = parlist->sublist("Problem").get("Fraction of nonzero weights", 0.5);
    fnzw = 1.0 - 2.0*fnzw;

    /*** Build vectors and dress them up as ROL vectors. ***/
    std::shared_ptr<const Tpetra::Map<> > vecmap_u = data->getMatA()->getDomainMap();
    std::shared_ptr<const Tpetra::Map<> > vecmap_z = data->getMatB()->getDomainMap();
    std::shared_ptr<const Tpetra::Map<> > vecmap_c = data->getMatA()->getRangeMap();
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > p_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > w_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > wup_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > wlo_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > z_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > c_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_c, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > dw_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
    // Set all values to 1 in u, z and c.
    u_rcp->putScalar(1.0);
    p_rcp->putScalar(1.0);
    z_rcp->putScalar(1.0);
    c_rcp->putScalar(1.0);
    w_rcp->randomize();
    wlo_rcp->putScalar(0.0);
    wup_rcp->putScalar(1.0);
    // Randomize d vectors.
    du_rcp->randomize();
    dw_rcp->randomize();
    dz_rcp->randomize();
    // Create ROL::TpetraMultiVectors.
    std::shared_ptr<ROL::Vector<RealT> > up = std::make_shared<ROL::TpetraMultiVector<RealT>>(u_rcp);
    std::shared_ptr<ROL::Vector<RealT> > pp = std::make_shared<ROL::TpetraMultiVector<RealT>>(p_rcp);
    std::shared_ptr<ROL::Vector<RealT> > wp = std::make_shared<ROL::TpetraMultiVector<RealT>>(w_rcp);
    std::shared_ptr<ROL::Vector<RealT> > wlop = std::make_shared<ROL::TpetraMultiVector<RealT>>(wlo_rcp);
    std::shared_ptr<ROL::Vector<RealT> > wupp = std::make_shared<ROL::TpetraMultiVector<RealT>>(wup_rcp);
    std::shared_ptr<ROL::Vector<RealT> > zp = std::make_shared<ROL::TpetraMultiVector<RealT>>(z_rcp);
    std::shared_ptr<ROL::Vector<RealT> > cp = std::make_shared<ROL::TpetraMultiVector<RealT>>(c_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dup = std::make_shared<ROL::TpetraMultiVector<RealT>>(du_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dwp = std::make_shared<ROL::TpetraMultiVector<RealT>>(dw_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dzp = std::make_shared<ROL::TpetraMultiVector<RealT>>(dz_rcp);
    // Create ROL SimOpt vectors.
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    /*** Build objective function, constraint and reduced objective function. ***/
    wp->applyUnary(IsGreaterThan<RealT>(fnzw, 1.0, 0.0));
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj =
      std::make_shared<Objective_PDEOPT_Poisson<RealT>>(data, w_rcp, parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con =
      std::make_shared<EqualityConstraint_PDEOPT_Poisson<RealT>>(data, parlist);
    std::shared_ptr<ROL::Objective<RealT> > objReduced =
      std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, zp, pp);

    /*** Check functional interface. ***/
    obj->checkGradient(x,d,true,*outStream);
    obj->checkHessVec(x,d,true,*outStream);
    con->checkApplyJacobian(x,d,*up,true,*outStream);
    con->checkApplyAdjointHessian(x,*dup,d,x,true,*outStream);
    con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
    con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
    con->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);
    objReduced->checkGradient(*zp,*dzp,true,*outStream);
    objReduced->checkHessVec(*zp,*dzp,true,*outStream);

    RealT tol = 1e-8;
    zp->zero();
    con->solve(*cp, *up, *zp, tol);
    data->outputTpetraVector(u_rcp, "data.txt");
    data->outputTpetraVector(data->getVecF(), "sources.txt");

    data->setVecUd(u_rcp);
    data->zeroRHS();

    /***
         Solve source inversion optimization problem with prescribed sensor locations.
    ***/
    ROL::Algorithm<RealT> algo_tr("Trust Region",*parlist,false);
    zp->zero(); // set zero initial guess
    algo_tr.run(*zp, *objReduced, true, *outStream);
    con->solve(*cp, *up, *zp, tol);

    //ROL::Algorithm<RealT> algo_cs("Composite Step",*parlist,false);
    //x.zero(); // set zero initial guess
    //algo_cs.run(x, *cp, *obj, *con, true, *outStream);

    data->outputTpetraVector(u_rcp, "state.txt");
    data->outputTpetraVector(z_rcp, "control.txt");
    data->outputTpetraVector(w_rcp, "weights.txt");
    //std::cout << std::endl << "Sum of random 0/1 entries: " << wp->reduce(ROL::Elementwise::ReductionSum<RealT>()) << std::endl;
    //data->outputTpetraData();

    /***
         Solve OED problem to obtain sparse sensor locations.
    ***/
    std::vector<std::shared_ptr<Tpetra::MultiVector<> > > randvecs_rcp;
    std::vector<std::shared_ptr<ROL::Vector<RealT> > > randvecs;
    std::vector<std::shared_ptr<ROL::Vector<RealT> > > training_models;
    // Get number of random vectors for trace estimation.
    int numRandVecs = parlistOED->sublist("Problem").get("OED Number of random vectors", 1);
    for (int i=0; i<numRandVecs; ++i) {
      randvecs_rcp.push_back( std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true));
      //std::shared_ptr<Tpetra::MultiVector<> > rand01_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
      randvecs.push_back( std::make_shared<ROL::TpetraMultiVector<RealT>>(randvecs_rcp[i]));
      //std::shared_ptr<ROL::Vector<RealT> > rand01p = std::make_shared<ROL::TpetraMultiVector<RealT>>(rand01_rcp);
      randvecs_rcp[i]->randomize();
      randvecs[i]->applyUnary(IsGreaterThan<RealT>(0.0, 1.0, -1.0));
      std::string fname = "rand" + std::to_string(i) + ".txt";
      data->outputTpetraVector(randvecs_rcp[i], fname);
    }

    std::shared_ptr<MyInterfaceOED> oed =
      std::make_shared<MyInterfaceOED>(obj, con, up, up, zp, zp, cp, cp, up, up, randvecs, training_models);
    ROL::ExperimentDesignObjective<RealT> objOED(oed, parlistOED);
    ROL::Bounds<RealT> bconOED(wlop, wupp);
    w_rcp->putScalar(1e-2);
    *outStream << std::endl << "Checking OED objective gradient:" << std::endl;
    dwp->scale(1e-2);
    objOED.checkGradient(*wp,*dwp,true,*outStream);
    ROL::Algorithm<RealT> algo_tr_oed("Trust Region",*parlistOED,false);
    wp->zero(); // set zero initial guess
    algo_tr_oed.run(*wp, objOED, bconOED, true, *outStream);
    data->outputTpetraVector(w_rcp, "weightsOED.txt");

    /***
         Solve source inversion optimization problem with optimal sensor locations.
    ***/
    wp->applyUnary(IsGreaterThan<RealT>(1e-1, 1.0, 0.0));
    RealT numLocOED = wp->reduce(ROL::Elementwise::ReductionSum<RealT>());
    *outStream << std::endl << "Number of nonzero OED locations: " << numLocOED << std::endl;
    zp->zero(); // set zero initial guess
    ROL::Algorithm<RealT> algo_tr_optimal("Trust Region",*parlist,false);
    algo_tr_optimal.run(*zp, *objReduced, true, *outStream);
    con->solve(*cp, *up, *zp, tol);
    data->outputTpetraVector(u_rcp, "stateOED.txt");
    data->outputTpetraVector(z_rcp, "controlOED.txt");
    data->outputTpetraVector(w_rcp, "weightsOED.txt");

    /***
         Solve source inversion optimization problem with random sensor locations.
    ***/
    w_rcp->randomize();
    RealT numLocTotal = w_rcp->getGlobalLength();
    wp->applyUnary(IsGreaterThan<RealT>((1-2*numLocOED/numLocTotal), 1.0, 0.0));
    *outStream << std::endl << "Number of nonzero random locations: " << wp->reduce(ROL::Elementwise::ReductionSum<RealT>()) << std::endl;
    zp->zero(); // set zero initial guess
    ROL::Algorithm<RealT> algo_tr_random("Trust Region",*parlist,false);
    algo_tr_random.run(*zp, *objReduced, true, *outStream);
    con->solve(*cp, *up, *zp, tol);
    data->outputTpetraVector(u_rcp, "stateRandom.txt");
    data->outputTpetraVector(z_rcp, "controlRandom.txt");
    data->outputTpetraVector(w_rcp, "weightsRandom.txt");

  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;
}
