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
    \brief Shows how to solve the mother problem of PDE-constrained optimization:
*/

#include "Teuchos_Comm.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Version.hpp"

#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationProblem.hpp"

#include "ROL_ScaledTpetraMultiVector.hpp"
#include "ROL_ScaledStdVector.hpp"

#include "ROL_Bounds.hpp"
#include "ROL_AugmentedLagrangian.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Elementwise_Reduce.hpp"

#include <iostream>
#include <algorithm>

#include "data.hpp"
#include "filter.hpp"
#include "parametrized_objective.hpp"
#include "parametrized_constraint.hpp"
#include "volume_constraint.hpp"
#include "build_sampler.hpp"

#include <fenv.h>

typedef double RealT;

std::shared_ptr<Tpetra::MultiVector<> > createTpetraVector(const std::shared_ptr<const Tpetra::Map<> > &map) {
  return std::make_shared<Tpetra::MultiVector<>>(map, 1, true);
}

int main(int argc, char *argv[]) {
//  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
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
    std::string filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    std::string stoch_filename = "stochastic.xml";
    std::shared_ptr<Teuchos::ParameterList> stoch_parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( stoch_filename, stoch_parlist.ptr() );

    /*** Initialize main data structure. ***/
    std::shared_ptr<const Teuchos::Comm<int> > serial_comm = std::make_shared<Teuchos::SerialComm<int>>();
    std::shared_ptr<ElasticitySIMPOperators<RealT> > data
      = std::make_shared<ElasticitySIMPOperators<RealT>>(serial_comm, parlist, outStream);
    /*** Initialize density filter. ***/
    std::shared_ptr<DensityFilter<RealT> > filter
      = std::make_shared<DensityFilter<RealT>>(serial_comm, parlist, outStream);
    /*** Build vectors and dress them up as ROL vectors. ***/
    std::shared_ptr<const Tpetra::Map<> > vecmap_u = data->getDomainMapA();
    std::shared_ptr<const Tpetra::Map<> > vecmap_z = data->getCellMap();
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp      = createTpetraVector(vecmap_u);
    std::shared_ptr<Tpetra::MultiVector<> > z_rcp      = createTpetraVector(vecmap_z);
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp     = createTpetraVector(vecmap_u);
    std::shared_ptr<Tpetra::MultiVector<> > dw_rcp     = createTpetraVector(vecmap_u);
    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp     = createTpetraVector(vecmap_z);
    std::shared_ptr<Tpetra::MultiVector<> > dz2_rcp    = createTpetraVector(vecmap_z);
    std::shared_ptr<std::vector<RealT> >    vc_rcp     = std::make_shared<std::vector<RealT>>(1, 0);
    std::shared_ptr<std::vector<RealT> >    vc_lam_rcp = std::make_shared<std::vector<RealT>>(1, 0);
    std::shared_ptr<std::vector<RealT> >    vscale_rcp = std::make_shared<std::vector<RealT>>(1, 0);
    // Set all values to 1 in u, z.
    RealT one(1), two(2);
    u_rcp->putScalar(one);
    // Set z to gray solution.
    RealT volFrac = parlist->sublist("ElasticityTopoOpt").get<RealT>("Volume Fraction");
    z_rcp->putScalar(volFrac);
    // Set scaling vector for constraint
    RealT W = parlist->sublist("Geometry").get<RealT>("Width");
    RealT H = parlist->sublist("Geometry").get<RealT>("Height");
    (*vscale_rcp)[0] = one/std::pow(W*H*(one-volFrac),two);
    // Set Scaling vector for density
    bool  useZscale = parlist->sublist("Problem").get<bool>("Use Scaled Density Vectors");
    RealT densityScaling = parlist->sublist("Problem").get<RealT>("Density Scaling");
    std::shared_ptr<Tpetra::MultiVector<> > scaleVec = createTpetraVector(vecmap_z);
    scaleVec->putScalar(densityScaling);
    if ( !useZscale ) {
      scaleVec->putScalar(one);
    }
    std::shared_ptr<const Tpetra::Vector<> > zscale_rcp = scaleVec->getVector(0);

    // Randomize d vectors.
    du_rcp->randomize(); //du_rcp->scale(0);
    dw_rcp->randomize();
    dz_rcp->randomize(); //dz_rcp->scale(0);
    dz2_rcp->randomize();
    // Create ROL::TpetraMultiVectors.
    std::shared_ptr<ROL::Vector<RealT> > up
      = std::make_shared<ROL::TpetraMultiVector<RealT>>(u_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dup
      = std::make_shared<ROL::TpetraMultiVector<RealT>>(du_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dwp
      = std::make_shared<ROL::TpetraMultiVector<RealT>>(dw_rcp);
    std::shared_ptr<ROL::Vector<RealT> > zp 
      = std::make_shared<ROL::PrimalScaledTpetraMultiVector<RealT>>(z_rcp,zscale_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dzp
      = std::make_shared<ROL::PrimalScaledTpetraMultiVector<RealT>>(dz_rcp,zscale_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dz2p
      = std::make_shared<ROL::PrimalScaledTpetraMultiVector<RealT>>(dz2_rcp,zscale_rcp);
    std::shared_ptr<ROL::Vector<RealT> > vcp
      = std::make_shared<ROL::PrimalScaledStdVector<RealT>>(vc_rcp,vscale_rcp);
    std::shared_ptr<ROL::Vector<RealT> > vc_lamp
      = std::make_shared<ROL::DualScaledStdVector<RealT>>(vc_lam_rcp,vscale_rcp);
    // Create ROL SimOpt vectors.
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);
    ROL::Vector_SimOpt<RealT> d2(dwp,dz2p);

    /*** Build sampler. ***/
    BuildSampler<RealT> buildSampler(comm,*stoch_parlist,*parlist);
    buildSampler.print("samples");

    /*** Compute compliance objective function scaling. ***/
    RealT min(ROL::ROL_INF<RealT>()), gmin(0), max(0), gmax(0), sum(0), gsum(0), tmp(0);
    Teuchos::Array<RealT> dotF(1, 0);
    RealT minDensity = parlist->sublist("ElasticitySIMP").get<RealT>("Minimum Density");
    for (int i = 0; i < buildSampler.get()->numMySamples(); ++i) {
      data->updateF(buildSampler.get()->getMyPoint(i));
      (data->getVecF())->dot(*(data->getVecF()),dotF.view(0,1));
      tmp = minDensity/dotF[0];
      min = ((min < tmp) ? min : tmp);
      max = ((max > tmp) ? max : tmp);
      sum += buildSampler.get()->getMyWeight(i) * tmp;
    }
    ROL::Elementwise::ReductionMin<RealT> ROLmin;
    buildSampler.getBatchManager()->reduceAll(&min,&gmin,1,ROLmin);
    ROL::Elementwise::ReductionMax<RealT> ROLmax;
    buildSampler.getBatchManager()->reduceAll(&max,&gmax,1,ROLmax);
    buildSampler.getBatchManager()->sumAll(&sum,&gsum,1);
    bool useExpValScale = stoch_parlist->sublist("Problem").get("Use Expected Value Scaling",false);
    RealT scale = (useExpValScale ? gsum : gmin);

    /*** Build objective function, constraint and reduced objective function. ***/
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj
       = std::make_shared<ParametrizedObjective_PDEOPT_ElasticitySIMP<RealT>>(data, filter, parlist,scale);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
       = std::make_shared<ParametrizedEqualityConstraint_PDEOPT_ElasticitySIMP<RealT>>(data, filter, parlist);
    std::shared_ptr<ROL::Reduced_Objective_SimOpt<RealT> > objReduced
       = std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, zp, dwp);
    std::shared_ptr<ROL::Constraint<RealT> > volcon
       = std::make_shared<EqualityConstraint_PDEOPT_ElasticitySIMP_Volume<RealT>>(data, parlist);

    /*** Build bound constraint ***/
    std::shared_ptr<Tpetra::MultiVector<> > lo_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
    std::shared_ptr<Tpetra::MultiVector<> > hi_rcp = std::make_shared<Tpetra::MultiVector<>>(vecmap_z, 1, true);
    lo_rcp->putScalar(0.0); hi_rcp->putScalar(1.0);
    std::shared_ptr<ROL::Vector<RealT> > lop
      = std::make_shared<ROL::PrimalScaledTpetraMultiVector<RealT>>(lo_rcp, zscale_rcp);
    std::shared_ptr<ROL::Vector<RealT> > hip
      = std::make_shared<ROL::PrimalScaledTpetraMultiVector<RealT>>(hi_rcp, zscale_rcp);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd = std::make_shared<ROL::Bounds<RealT>>(lop,hip);

    /*** Build Stochastic Functionality. ***/
    ROL::OptimizationProblem<RealT> opt(objReduced,zp,bnd);
    opt.setStochasticObjective(*stoch_parlist,buildSampler.get());

    /*** Check functional interface. ***/
    bool checkDeriv = parlist->sublist("Problem").get("Derivative Check",false);
    if ( checkDeriv ) {
//      *outStream << "Checking Objective:" << "\n";
//      obj->checkGradient(x,d,true,*outStream);
//      obj->checkHessVec(x,d,true,*outStream);
//      obj->checkHessSym(x,d,d2,true,*outStream);
//      *outStream << "Checking Constraint:" << "\n";
//      con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
//      con->checkAdjointConsistencyJacobian_1(*dwp, *dup, *up, *zp, true, *outStream);
//      con->checkAdjointConsistencyJacobian_2(*dwp, *dzp, *up, *zp, true, *outStream);
//      con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
//      con->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);
//      con->checkApplyJacobian(x,d,*up,true,*outStream);
//      con->checkApplyAdjointHessian(x,*dup,d,x,true,*outStream);
      *outStream << "Checking Reduced Objective:" << "\n";
      opt.check(*outStream);
      *outStream << "Checking Volume Constraint:" << "\n";
      volcon->checkAdjointConsistencyJacobian(*vcp,*dzp,*zp,true,*outStream);
      volcon->checkApplyJacobian(*zp,*dzp,*vcp,true,*outStream);
      volcon->checkApplyAdjointHessian(*zp,*vcp,*dzp,*zp,true,*outStream);
    }

    /*** Run optimization ***/
    ROL::AugmentedLagrangian<RealT> augLag(opt.getObjective(),volcon,*vc_lamp,1.0,
                                          *opt.getSolutionVector(),*vcp,*parlist);
    ROL::Algorithm<RealT> algo("Augmented Lagrangian",*parlist,false);
    std::clock_t timer = std::clock();
    algo.run(*opt.getSolutionVector(),*vc_lamp,augLag,*volcon,*opt.getBoundConstraint(),true,*outStream);
    *outStream << "Optimization time: "
               << static_cast<RealT>(std::clock()-timer)/static_cast<RealT>(CLOCKS_PER_SEC)
               << " seconds." << std::endl;

    data->outputTpetraVector(z_rcp, "density.txt");
    data->outputTpetraVector(u_rcp, "state.txt");
    data->outputTpetraVector(zscale_rcp, "weights.txt");

    // Build objective function distribution
    RealT val(0);
    int nSamp = stoch_parlist->sublist("Problem").get("Number of Output Samples",10);
    stoch_parlist->sublist("Problem").set("Number of Samples",nSamp);
    BuildSampler<RealT> buildSampler_dist(comm,*stoch_parlist,*parlist);
    std::stringstream name;
    name << "samples_" << buildSampler_dist.getBatchManager()->batchID() << ".txt";
    std::ofstream file;
    file.open(name.str());
    std::vector<RealT> sample;
    RealT tol = 1.e-8;
    std::clock_t timer_print = std::clock();
    for (int i = 0; i < buildSampler_dist.get()->numMySamples(); ++i) {
      sample = buildSampler_dist.get()->getMyPoint(i);
      objReduced->setParameter(sample);
      val = objReduced->value(*zp,tol);
      for (int j = 0; j < static_cast<int>(sample.size()); ++j) {
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
