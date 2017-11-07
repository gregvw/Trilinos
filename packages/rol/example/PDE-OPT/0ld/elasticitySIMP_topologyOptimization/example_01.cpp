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

#include "ROL_Algorithm.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_CompositeStep.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_ScaledTpetraMultiVector.hpp"

#include "ROL_ScaledStdVector.hpp"

#include "ROL_Bounds.hpp"
#include "ROL_AugmentedLagrangian.hpp"
#include "ROL_Algorithm.hpp"

#include <iostream>
#include <algorithm>

#include "data.hpp"
#include "filter.hpp"
#include "objective.hpp"
#include "constraint.hpp"
#include "volume_constraint.hpp"

//#include <fenv.h>

typedef double RealT;

std::shared_ptr<Tpetra::MultiVector<> > createTpetraVector(const std::shared_ptr<const Tpetra::Map<> > &map) {
  return std::make_shared<Tpetra::MultiVector<>>(map, 1, true);
}

int main(int argc, char *argv[]) {
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

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

    /*** Initialize main data structure. ***/
    std::shared_ptr<ElasticitySIMPOperators<RealT> > data
      = std::make_shared<ElasticitySIMPOperators<RealT>>(comm, parlist, outStream);
    /*** Initialize density filter. ***/
    std::shared_ptr<DensityFilter<RealT> > filter
      = std::make_shared<DensityFilter<RealT>>(comm, parlist, outStream);
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
    u_rcp->putScalar(1.0);
    // Set z to gray solution.
    RealT volFrac = parlist->sublist("ElasticityTopoOpt").get<RealT>("Volume Fraction");
    z_rcp->putScalar(volFrac);
    // Set scaling vector for constraint
    RealT W = parlist->sublist("Geometry").get<RealT>("Width");
    RealT H = parlist->sublist("Geometry").get<RealT>("Height");
    RealT one(1), two(2);
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

   //test     
   /*data->updateMaterialDensity (z_rcp);
    std::shared_ptr<Tpetra::MultiVector<RealT> > rhs
      = std::make_shared<Tpetra::MultiVector<> (data->getVecF()->getMap>(), 1, true);
    data->ApplyMatAToVec(rhs, u_rcp);
    data->outputTpetraVector(rhs, "KU0.txt");
    data->ApplyInverseJacobian1ToVec(u_rcp, rhs, false);
    data->outputTpetraVector(u_rcp, "KKU0.txt");
    
    data->ApplyJacobian1ToVec(rhs, u_rcp);
    data->outputTpetraVector(rhs, "KU1.txt");
    data->ApplyInverseJacobian1ToVec(u_rcp, rhs, false);
    data->outputTpetraVector(u_rcp, "KKU1.txt");
  */
    //u_rcp->putScalar(1.0);
    //z_rcp->putScalar(1.0);
    // Randomize d vectors.
    du_rcp->randomize(); //du_rcp->scale(0);
    dw_rcp->randomize();
    dz_rcp->randomize(); //dz_rcp->scale(0);
    dz2_rcp->randomize();
    // Create ROL::TpetraMultiVectors.
    std::shared_ptr<ROL::Vector<RealT> > up   = std::make_shared<ROL::TpetraMultiVector<RealT>>(u_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dup  = std::make_shared<ROL::TpetraMultiVector<RealT>>(du_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dwp  = std::make_shared<ROL::TpetraMultiVector<RealT>>(dw_rcp);
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

    /*** Build objective function, constraint and reduced objective function. ***/
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj
       = std::make_shared<Objective_PDEOPT_ElasticitySIMP<RealT>>(data, filter, parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
       = std::make_shared<EqualityConstraint_PDEOPT_ElasticitySIMP<RealT>>(data, filter, parlist);
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

    /*** Check functional interface. ***/
    *outStream << "Checking Objective:" << "\n";
    obj->checkGradient(x,d,true,*outStream);
    obj->checkHessVec(x,d,true,*outStream);
    obj->checkHessSym(x,d,d2,true,*outStream);
    *outStream << "Checking Constraint:" << "\n";
    con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
    con->checkAdjointConsistencyJacobian_1(*dwp, *dup, *up, *zp, true, *outStream);
    con->checkAdjointConsistencyJacobian_2(*dwp, *dzp, *up, *zp, true, *outStream);
    con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
    con->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);
    con->checkApplyJacobian(x,d,*up,true,*outStream);
    con->checkApplyAdjointHessian(x,*dup,d,x,true,*outStream);
    *outStream << "Checking Reduced Objective:" << "\n";
    objReduced->checkGradient(*zp,*dzp,true,*outStream);
    objReduced->checkHessVec(*zp,*dzp,true,*outStream);
    *outStream << "Checking Volume Constraint:" << "\n";
    volcon->checkAdjointConsistencyJacobian(*vcp,*dzp,*zp,true,*outStream);
    volcon->checkApplyJacobian(*zp,*dzp,*vcp,true,*outStream);
    volcon->checkApplyAdjointHessian(*zp,*vcp,*dzp,*zp,true,*outStream);

    /*** Run optimization ***/
    ROL::AugmentedLagrangian<RealT> augLag(objReduced,volcon,*vc_lamp,1.0,*zp,*vcp,*parlist);
    ROL::Algorithm<RealT> algo("Augmented Lagrangian",*parlist,false);
    algo.run(*zp,*vc_lamp,augLag,*volcon,*bnd,true,*outStream);
    //ROL::MoreauYosidaPenalty<RealT> MYpen(objReduced,bnd,*zp,*parlist);
    //ROL::Algorithm<RealT> algo("Moreau-Yosida Penalty",*parlist,false);
    //algo.run(*zp,*vc_lamp,MYpen,*volcon,*bnd,true,*outStream);

    // new filter, for testing
    /*parlist->sublist("Density Filter").set("Enable", true);
    std::shared_ptr<DensityFilter<RealT> > testfilter
      = std::make_shared<DensityFilter<RealT>>(comm, parlist, outStream);
    std::shared_ptr<Tpetra::MultiVector<> > z_filtered_rcp = std::make_shared<Tpetra::MultiVector<>>(*z_rcp, Teuchos::Copy);
    testfilter->apply(z_filtered_rcp, z_rcp);
    std::shared_ptr<Tpetra::MultiVector<> > cm_rcp = data->getCellAreas();
    std::shared_ptr<Tpetra::MultiVector<> > icm_rcp = std::make_shared<Tpetra::MultiVector<>>(*cm_rcp, Teuchos::Copy);
    std::shared_ptr<Tpetra::MultiVector<> > zf_scaled_rcp = std::make_shared<Tpetra::MultiVector<>>(*z_rcp, Teuchos::Copy);
    icm_rcp->reciprocal(*cm_rcp);
    zf_scaled_rcp->elementWiseMultiply(1.0, *(icm_rcp->getVector(0)), *z_filtered_rcp, 0.0);
    data->outputTpetraVector(zf_scaled_rcp, "density_filtered_scaled.txt");
    data->outputTpetraVector(z_filtered_rcp, "density_filtered.txt");*/
    
    data->outputTpetraVector(z_rcp, "density.txt");
    data->outputTpetraVector(u_rcp, "state.txt");
    data->outputTpetraVector(zscale_rcp, "weights.txt");

    // Get a summary from the time monitor.
    Teuchos::TimeMonitor::summarize();
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
