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
           advection-diffusion equation.
*/

#include "Teuchos_Comm.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Version.hpp"

#include "ROL_Algorithm.hpp"
#include "ROL_TpetraMultiVector.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_OptimizationProblem.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_StdTeuchosBatchManager.hpp"
#include "ROL_TpetraTeuchosBatchManager.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Bounds.hpp"

#include <iostream>
#include <algorithm>

#include "data.hpp"
#include "objective.hpp"
#include "constraint.hpp"

typedef double RealT;

template<class Real>
Real random(const std::shared_ptr<const Teuchos::Comm<int> > &commptr) {
  Real val = 0.0;
  if ( Teuchos::rank<int>(*commptr)==0 ) {
    val = (Real)rand()/(Real)RAND_MAX;
  }
  Teuchos::broadcast<int,Real>(*commptr,0,1,&val);
  return val;
}

template<class Real>
void randomize(std::vector<Real> &x,
               const std::shared_ptr<const Teuchos::Comm<int> > &commptr) {
  unsigned dim = x.size();
  for ( unsigned i = 0; i < dim; i++ ) {
    x[i] = random<Real>(commptr);
  }
}

int main(int argc, char *argv[]) {

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  /*** Initialize communicator. ***/
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &bhs);
  std::shared_ptr<const Teuchos::Comm<int> > comm
    = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  std::shared_ptr<const Teuchos::Comm<int> > serial_comm
    = std::make_shared<Teuchos::SerialComm<int>>();
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
    std::string filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );

    /*** Initialize main data structure. ***/
    std::shared_ptr<PoissonData<RealT> > data
      = std::make_shared<PoissonData<RealT>>(serial_comm, parlist, outStream);

    /*** Build vectors and dress them up as ROL vectors. ***/
    const RealT zero(0), one(1);
    std::shared_ptr<const Tpetra::Map<> > vecmap_u = data->getMatA()->getDomainMap();
//    std::shared_ptr<const Tpetra::Map<> > vecmap_z = data->getMatB()->getDomainMap();
    std::shared_ptr<const Tpetra::Map<> > vecmap_c = data->getMatA()->getRangeMap();
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
//    std::shared_ptr<Tpetra::MultiVector<> > z_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > p_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > c_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_c, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_u, 1, true);
//    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
    std::shared_ptr<std::vector<RealT> > z_rcp  = std::make_shared<std::vector<RealT>>(9,one);
    std::shared_ptr<std::vector<RealT> > dz_rcp = std::make_shared<std::vector<RealT>>(9,zero);
    // Set all values to 1 in u, z and c.
    u_rcp->putScalar(one);
//    z_rcp->putScalar(one);
    p_rcp->putScalar(one);
    c_rcp->putScalar(one);
    // Randomize d vectors.
    du_rcp->randomize();
    //dz_rcp->randomize();
    randomize<RealT>(*dz_rcp,comm);
    // Create ROL::TpetraMultiVectors.
    std::shared_ptr<ROL::Vector<RealT> > up = std::make_shared<ROL::TpetraMultiVector<RealT>>(u_rcp);
//    std::shared_ptr<ROL::Vector<RealT> > zp = std::make_shared<ROL::TpetraMultiVector<RealT>>(z_rcp);
    std::shared_ptr<ROL::Vector<RealT> > pp = std::make_shared<ROL::TpetraMultiVector<RealT>>(p_rcp);
    std::shared_ptr<ROL::Vector<RealT> > cp = std::make_shared<ROL::TpetraMultiVector<RealT>>(c_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dup = std::make_shared<ROL::TpetraMultiVector<RealT>>(du_rcp);
//    std::shared_ptr<ROL::Vector<RealT> > dzp = std::make_shared<ROL::TpetraMultiVector<RealT>>(dz_rcp);
    std::shared_ptr<ROL::Vector<RealT> > zp = std::make_shared<ROL::StdVector<RealT>>(z_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dzp = std::make_shared<ROL::StdVector<RealT>>(dz_rcp);
    // Create ROL SimOpt vectors.
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    /*** Build objective function, constraint and reduced objective function. ***/
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj
      = std::make_shared<Objective_PDEOPT_Poisson<RealT>>(data, parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
      = std::make_shared<EqualityConstraint_PDEOPT_Poisson<RealT>>(data, parlist);
    std::shared_ptr<ROL::Reduced_Objective_SimOpt<RealT> > objReduced
      = std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, zp, pp);
    std::shared_ptr<std::vector<RealT> > zlo_rcp = std::make_shared<std::vector<RealT>>(9,zero);
    std::shared_ptr<std::vector<RealT> > zup_rcp = std::make_shared<std::vector<RealT>>(9,one);
    std::shared_ptr<ROL::Vector<RealT> > zlop = std::make_shared<ROL::StdVector<RealT>>(zlo_rcp);
    std::shared_ptr<ROL::Vector<RealT> > zupp = std::make_shared<ROL::StdVector<RealT>>(zup_rcp);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd
      = std::make_shared<ROL::Bounds<RealT>>(zlop,zupp);

    /*** Build sampler ***/
    int sdim  = 37;
    int nsamp = parlist->sublist("Problem").get("Number of Samples",100);
    std::vector<RealT> tmp = {-one, one};
    std::vector<std::vector<RealT> > bounds(sdim,tmp);
    std::shared_ptr<ROL::BatchManager<RealT> > bman
      = std::make_shared<ROL::StdTeuchosBatchManager<RealT,int>>(comm);
      //= std::make_shared<ROL::TpetraTeuchosBatchManager<RealT>>(comm);
      //= std::make_shared<ROL::BatchManager<RealT>>();
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler
      = std::make_shared<ROL::MonteCarloGenerator<RealT>>(nsamp,bounds,bman);
    // Build stochastic problem
    ROL::OptimizationProblem<RealT> opt(objReduced,zp,bnd);
    parlist->sublist("SOL").set("Initial Statistic",zero);
    opt.setStochasticObjective(*parlist,sampler);

    bool printMeanValueState = parlist->sublist("Problem").get("Print Mean Value State",false);
    if ( printMeanValueState ) {
      RealT tol = 1.e-8;
      std::vector<RealT> my_sample(sdim), mev_sample(sdim), gev_sample(sdim);
      for (int i = 0; i < sampler->numMySamples(); ++i) {
        my_sample = sampler->getMyPoint(i);
        for (int j = 0; j < sdim; ++j) {
          mev_sample[j] += sampler->getMyWeight(i)*my_sample[j];
        }
      }
      bman->sumAll(&mev_sample[0],&gev_sample[0],sdim);
      con->setParameter(gev_sample);
      zp->zero();
      con->solve(*cp,*up,*zp,tol);
      data->outputTpetraVector(u_rcp, "mean_value_state.txt");
    }

    /*** Check functional interface. ***/
    bool checkDeriv = parlist->sublist("Problem").get("Check Derivatives",false);
    if ( checkDeriv ) {
      std::vector<RealT> param(sdim,1);
      objReduced->setParameter(param);
      obj->checkGradient(x,d,true,*outStream);
      obj->checkHessVec(x,d,true,*outStream);
      con->checkApplyJacobian(x,d,*up,true,*outStream);
      con->checkApplyAdjointHessian(x,*dup,d,x,true,*outStream);
      con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
      con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
      con->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);
      objReduced->checkGradient(*zp,*dzp,true,*outStream);
      objReduced->checkHessVec(*zp,*dzp,true,*outStream);
      opt.check(*outStream);
    }

    /*** Solve optimization problem. ***/
    std::shared_ptr<ROL::Algorithm<RealT> > algo;
    bool useBundle = parlist->sublist("Problem").get("Is problem nonsmooth?",false);
    if ( useBundle ) {
      algo = std::make_shared<ROL::Algorithm<RealT>>("Bundle",*parlist,false);
    }
    else {
      algo = std::make_shared<ROL::Algorithm<RealT>>("Trust Region",*parlist,false);
    }
    zp->zero(); // set zero initial guess
    std::clock_t timer = std::clock();
    algo->run(opt,true,*outStream);
    *outStream << "Optimization time: "
               << static_cast<RealT>(std::clock()-timer)/static_cast<RealT>(CLOCKS_PER_SEC)
               << " seconds." << std::endl;

    // Output control to file
    //data->outputTpetraVector(z_rcp, "control.txt");
    std::clock_t timer_print = std::clock();
    if ( myRank == 0 ) {
      std::ofstream zfile;
      zfile.open("control.txt");
      for (int i = 0; i < 9; i++) {
        zfile << (*z_rcp)[i] << "\n";
      }
      zfile.close();
    }

    // Output expected state to file
    up->zero(); pp->zero(); dup->zero();
    RealT tol(1.e-8);
    std::shared_ptr<ROL::BatchManager<RealT> > bman_Eu
      = std::make_shared<ROL::TpetraTeuchosBatchManager<RealT>>(comm);
    std::vector<RealT> sample(sdim);
    std::stringstream name_samp;
    name_samp << "samples_" << bman->batchID() << ".txt";
    std::ofstream file_samp;
    file_samp.open(name_samp.str());
    for (int i = 0; i < sampler->numMySamples(); ++i) {
      sample = sampler->getMyPoint(i);
      con->setParameter(sample);
      con->solve(*cp,*dup,*zp,tol);
      up->axpy(sampler->getMyWeight(i),*dup);
      for (int j = 0; j < sdim; ++j) {
        file_samp << sample[j] << "  ";
      }
      file_samp << "\n";
    }
    file_samp.close();
    bman_Eu->sumAll(*up,*pp);
    data->outputTpetraVector(p_rcp, "mean_state.txt");

    // Build objective function distribution
    RealT val(0);
    int nsamp_dist = parlist->sublist("Problem").get("Number of Output Samples",100);
      //= std::make_shared<ROL::TpetraTeuchosBatchManager<RealT>>(comm);
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler_dist
      = std::make_shared<ROL::MonteCarloGenerator<RealT>>(nsamp_dist,bounds,bman);
    std::stringstream name;
    name << "obj_samples_" << bman->batchID() << ".txt";
    std::ofstream file;
    file.open(name.str());
    for (int i = 0; i < sampler_dist->numMySamples(); ++i) {
      sample = sampler_dist->getMyPoint(i);
      objReduced->setParameter(sample);
      val = objReduced->value(*zp,tol);
      for (int j = 0; j < sdim; ++j) {
        file << sample[j] << "  ";
      }
      file << val << "\n";
    }
    file.close();
    *outStream << "Output time: "
               << static_cast<RealT>(std::clock()-timer_print)/static_cast<RealT>(CLOCKS_PER_SEC)
               << " seconds." << std::endl;
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
