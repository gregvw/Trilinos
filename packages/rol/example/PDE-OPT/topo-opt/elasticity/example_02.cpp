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
    \brief Shows how to solve the stuctural topology optimization problem.
*/

#include "Teuchos_Comm.hpp"
#include "Teuchos_Time.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Version.hpp"

#include <iostream>
#include <algorithm>

#include "ROL_OptimizationSolver.hpp"
#include "ROL_OptimizationProblem.hpp"
#include "ROL_ScaledStdVector.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"
#include "ROL_CompositeConstraint_SimOpt.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_TpetraTeuchosBatchManager.hpp"
#include "ROL_Bounds.hpp"

#include "../../TOOLS/pdeconstraint.hpp"
#include "../../TOOLS/linearpdeconstraint.hpp"
#include "../../TOOLS/pdeobjective.hpp"
#include "../../TOOLS/pdevector.hpp"
#include "../../TOOLS/integralconstraint.hpp"
#include "obj_topo-opt.hpp"
#include "mesh_topo-opt.hpp"
#include "pde_elasticity.hpp"
#include "pde_filter.hpp"

typedef double RealT;

int main(int argc, char *argv[]) {
  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  ROL::SharedPointer<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  /*** Initialize communicator. ***/
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &bhs);
  ROL::SharedPointer<const Teuchos::Comm<int> > comm
    = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  ROL::SharedPointer<const Teuchos::Comm<int> > serial_comm
    = ROL::makeShared<Teuchos::SerialComm<int>>();
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
    RealT tol(1e-8), one(1);

    /*** Read in XML input ***/
    std::string filename = "input_ex02.xml";
    ROL::SharedPointer<Teuchos::ParameterList> parlist = ROL::makeShared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );

    // Retrieve parameters.
    const RealT domainWidth  = parlist->sublist("Geometry").get("Width", 1.0);
    const RealT domainHeight = parlist->sublist("Geometry").get("Height", 1.0);
    const RealT volFraction  = parlist->sublist("Problem").get("Volume Fraction", 0.4);
    const RealT objFactor    = parlist->sublist("Problem").get("Objective Scaling", 1e-4);

    /*** Initialize main data structure. ***/
    ROL::SharedPointer<MeshManager<RealT> > meshMgr
      = ROL::makeShared<MeshManager_TopoOpt<RealT>>(*parlist);
    // Initialize PDE describing elasticity equations.
    ROL::SharedPointer<PDE_Elasticity<RealT> > pde
      = ROL::makeShared<PDE_Elasticity<RealT>>(*parlist);
    ROL::SharedPointer<ROL::Constraint_SimOpt<RealT> > con
      = ROL::makeShared<PDE_Constraint<RealT>>(pde,meshMgr,serial_comm,*parlist,*outStream);
    // Initialize the filter PDE.
    ROL::SharedPointer<PDE_Filter<RealT> > pdeFilter
      = ROL::makeShared<PDE_Filter<RealT>>(*parlist);
    ROL::SharedPointer<ROL::Constraint_SimOpt<RealT> > conFilter
      = ROL::makeShared<Linear_PDE_Constraint<RealT>>(pdeFilter,meshMgr,serial_comm,*parlist,*outStream);
    // Cast the constraint and get the assembler.
    ROL::SharedPointer<PDE_Constraint<RealT> > pdecon
      = ROL::dynamicPointerCast<PDE_Constraint<RealT> >(con);
    ROL::SharedPointer<Assembler<RealT> > assembler = pdecon->getAssembler();
    con->setSolveParameters(*parlist);

    // Create state vector.
    ROL::SharedPointer<Tpetra::MultiVector<> > u_rcp = assembler->createStateVector();
    u_rcp->randomize();
    ROL::SharedPointer<ROL::Vector<RealT> > up
      = ROL::makeShared<PDE_PrimalSimVector<RealT>>(u_rcp,pde,assembler,*parlist);
    ROL::SharedPointer<Tpetra::MultiVector<> > p_rcp = assembler->createStateVector();
    p_rcp->randomize();
    ROL::SharedPointer<ROL::Vector<RealT> > pp
      = ROL::makeShared<PDE_PrimalSimVector<RealT>>(p_rcp,pde,assembler,*parlist);
    // Create control vector.
    ROL::SharedPointer<Tpetra::MultiVector<> > z_rcp = assembler->createControlVector();
    //z_rcp->randomize();
    z_rcp->putScalar(volFraction);
    //z_rcp->putScalar(0);
    ROL::SharedPointer<ROL::Vector<RealT> > zp
      = ROL::makeShared<PDE_PrimalOptVector<RealT>>(z_rcp,pde,assembler,*parlist);
    // Create residual vector.
    ROL::SharedPointer<Tpetra::MultiVector<> > r_rcp = assembler->createResidualVector();
    r_rcp->putScalar(0.0);
    ROL::SharedPointer<ROL::Vector<RealT> > rp
      = ROL::makeShared<PDE_DualSimVector<RealT>>(r_rcp,pde,assembler,*parlist);
    // Create state direction vector.
    ROL::SharedPointer<Tpetra::MultiVector<> > du_rcp = assembler->createStateVector();
    du_rcp->randomize();
    //du_rcp->putScalar(0);
    ROL::SharedPointer<ROL::Vector<RealT> > dup
      = ROL::makeShared<PDE_PrimalSimVector<RealT>>(du_rcp,pde,assembler,*parlist);
    // Create control direction vector.
    ROL::SharedPointer<Tpetra::MultiVector<> > dz_rcp = assembler->createControlVector();
    dz_rcp->randomize();
    dz_rcp->scale(0.01);
    ROL::SharedPointer<ROL::Vector<RealT> > dzp
      = ROL::makeShared<PDE_PrimalOptVector<RealT>>(dz_rcp,pde,assembler,*parlist);
    // Create control test vector.
    ROL::SharedPointer<Tpetra::MultiVector<> > rz_rcp = assembler->createControlVector();
    rz_rcp->randomize();
    ROL::SharedPointer<ROL::Vector<RealT> > rzp
      = ROL::makeShared<PDE_PrimalOptVector<RealT>>(rz_rcp,pde,assembler,*parlist);

    ROL::SharedPointer<Tpetra::MultiVector<> > dualu_rcp = assembler->createStateVector();
    ROL::SharedPointer<ROL::Vector<RealT> > dualup
      = ROL::makeShared<PDE_DualSimVector<RealT>>(dualu_rcp,pde,assembler,*parlist);
    ROL::SharedPointer<Tpetra::MultiVector<> > dualz_rcp = assembler->createControlVector();
    ROL::SharedPointer<ROL::Vector<RealT> > dualzp
      = ROL::makeShared<PDE_DualOptVector<RealT>>(dualz_rcp,pde,assembler,*parlist);

    // Create ROL SimOpt vectors.
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    // Initialize "filtered" of "unfiltered" constraint.
    ROL::SharedPointer<ROL::Constraint_SimOpt<RealT> > pdeWithFilter;
    bool useFilter  = parlist->sublist("Problem").get("Use Filter", true);
    if (useFilter) {
      pdeWithFilter = ROL::makeShared<ROL::CompositeConstraint_SimOpt<RealT>>(con, conFilter, *rp, *rp, *up, *zp, *zp);
    }
    else {
      pdeWithFilter = con;
    }
    pdeWithFilter->setSolveParameters(*parlist);

    // Initialize compliance objective function.
    Teuchos::ParameterList list(*parlist);
    list.sublist("Vector").sublist("Sim").set("Use Riesz Map",true);
    list.sublist("Vector").sublist("Sim").set("Lump Riesz Map",false);
    // Has state Riesz map enabled for mesh-independent compliance scaling.
    ROL::SharedPointer<Tpetra::MultiVector<> > f_rcp = assembler->createResidualVector();
    f_rcp->putScalar(0.0);
    ROL::SharedPointer<ROL::Vector<RealT> > fp
      = ROL::makeShared<PDE_DualSimVector<RealT>>(f_rcp,pde,assembler,list);
    up->zero();
    con->value(*fp, *up, *zp, tol);
    RealT objScaling = objFactor, fnorm2 = fp->dot(*fp);
    if (fnorm2 > 1e2*ROL::ROL_EPSILON<RealT>()) {
      objScaling /= fnorm2;
    }
    u_rcp->randomize();
    std::vector<ROL::SharedPointer<QoI<RealT> > > qoi_vec(1,ROL::nullPointer);
    qoi_vec[0] = ROL::makeShared<QoI_TopoOpt<RealT>(pde->getFE(>(),
                                                     pde->getLoad(),
                                                     pde->getFieldHelper(),
                                                     objScaling));
    ROL::SharedPointer<ROL::Objective_SimOpt<RealT> > obj
      = ROL::makeShared<PDE_Objective<RealT>>(qoi_vec,assembler);

    // Initialize volume constraint,
    ROL::SharedPointer<QoI<RealT> > qoi_vol
      = ROL::makeShared<QoI_Volume_TopoOpt<RealT>(pde->getFE(),pde->getFieldHelper>(),*parlist);
    ROL::SharedPointer<IntegralOptConstraint<RealT> > vcon
      = ROL::makeShared<IntegralOptConstraint<RealT>>(qoi_vol,assembler);
    // Create volume constraint vector and set to zero
    RealT vecScaling = one / std::pow(domainWidth*domainHeight*(one-volFraction), 2);
    ROL::SharedPointer<std::vector<RealT> > scalevec_rcp = ROL::makeShared<std::vector<RealT>>(1,vecScaling);
    ROL::SharedPointer<std::vector<RealT> > c1_rcp = ROL::makeShared<std::vector<RealT>>(1,0);
    ROL::SharedPointer<ROL::Vector<RealT> > c1p = ROL::makeShared<ROL::PrimalScaledStdVector<RealT>>(c1_rcp, scalevec_rcp);
    ROL::SharedPointer<std::vector<RealT> > c2_rcp = ROL::makeShared<std::vector<RealT>>(1,1);
    ROL::SharedPointer<ROL::Vector<RealT> > c2p = ROL::makeShared<ROL::DualScaledStdVector<RealT>>(c2_rcp, scalevec_rcp);

    // Initialize bound constraints.
    ROL::SharedPointer<Tpetra::MultiVector<> > lo_rcp = assembler->createControlVector();
    ROL::SharedPointer<Tpetra::MultiVector<> > hi_rcp = assembler->createControlVector();
    lo_rcp->putScalar(0.0); hi_rcp->putScalar(1.0);
    ROL::SharedPointer<ROL::Vector<RealT> > lop
      = ROL::makeShared<PDE_PrimalOptVector<RealT>>(lo_rcp,pde,assembler);
    ROL::SharedPointer<ROL::Vector<RealT> > hip
      = ROL::makeShared<PDE_PrimalOptVector<RealT>>(hi_rcp,pde,assembler);
    ROL::SharedPointer<ROL::BoundConstraint<RealT> > bnd
      = ROL::makeShared<ROL::Bounds<RealT>>(lop,hip);

    // Initialize Augmented Lagrangian functional.
    bool storage = parlist->sublist("Problem").get("Use state storage",true);
    ROL::SharedPointer<ROL::SimController<RealT> > stateStore
      = ROL::makeShared<ROL::SimController<RealT>>();
    ROL::SharedPointer<ROL::Reduced_Objective_SimOpt<RealT> > objRed
      = Teuchos::rcp(new
        ROL::Reduced_Objective_SimOpt<RealT>(obj,pdeWithFilter,
                                             stateStore,up,zp,pp,
                                             storage));

    /*************************************************************************/
    /***************** BUILD SAMPLER *****************************************/
    /*************************************************************************/
    int nsamp  = parlist->sublist("Problem").get("Number of samples", 4);
    Teuchos::Array<RealT> loadMag
      = Teuchos::getArrayFromStringParameter<double>(parlist->sublist("Problem").sublist("Load"), "Magnitude");
    int nLoads = loadMag.size();
    int dim    = 2;
    std::vector<ROL::SharedPointer<ROL::Distribution<RealT> > > distVec(dim*nLoads);
    for (int i = 0; i < nLoads; ++i) {
      std::stringstream sli;
      sli << "Stochastic Load " << i;
      Teuchos::ParameterList magList;
      magList.sublist("Distribution") = parlist->sublist("Problem").sublist(sli.str()).sublist("Magnitude");
      //magList.print(*outStream);
      distVec[i*dim + 0] = ROL::DistributionFactory<RealT>(magList);
      Teuchos::ParameterList angList;
      angList.sublist("Distribution") = parlist->sublist("Problem").sublist(sli.str()).sublist("Polar Angle");
      //angList.print(*outStream);
      distVec[i*dim + 1] = ROL::DistributionFactory<RealT>(angList);
    }
    //ROL::SharedPointer<ROL::BatchManager<RealT> > bman
    //  = ROL::makeShared<PDE_OptVector_BatchManager<RealT>>(comm);
    ROL::SharedPointer<ROL::BatchManager<RealT> > bman
      = ROL::makeShared<ROL::TpetraTeuchosBatchManager<RealT>>(comm);
    ROL::SharedPointer<ROL::SampleGenerator<RealT> > sampler
      = ROL::makeShared<ROL::MonteCarloGenerator<RealT>>(nsamp,distVec,bman);

    /*************************************************************************/
    /***************** BUILD STOCHASTIC PROBLEM ******************************/
    /*************************************************************************/
    ROL::OptimizationProblem<RealT> opt(objRed,zp,bnd,vcon,c2p);
    parlist->sublist("SOL").set("Initial Statistic",one);
    opt.setStochasticObjective(*parlist,sampler);

    //ROL::AugmentedLagrangian<RealT> augLag(opt.getObjective(),
    //                                       opt.getConstraint(),
    //                                       *c2p,
    //                                       1,
    //                                       *opt.getSolutionVector(),
    //                                       *c1p,
    //                                       *parlist);

    // Run derivative checks
    bool checkDeriv = parlist->sublist("Problem").get("Check derivatives",false);
    if ( checkDeriv ) {
      *outStream << "\n\nCheck Opt Vector\n";
      zp->checkVector(*dzp,*rzp,true,*outStream);

      *outStream << "\n\nCheck Gradient of Full Objective Function\n";
      obj->checkGradient(x,d,true,*outStream);
      *outStream << "\n\nCheck Hessian of Full Objective Function\n";
      obj->checkHessVec(x,d,true,*outStream);

      *outStream << "\n\nCheck Full Jacobian of PDE Constraint\n";
      con->checkApplyJacobian(x,d,*rp,true,*outStream);
      *outStream << "\n\nCheck Jacobian_1 of PDE Constraint\n";
      con->checkApplyJacobian_1(*up,*zp,*dup,*rp,true,*outStream);
      *outStream << "\n\nCheck Jacobian_2 of PDE Constraint\n";
      con->checkApplyJacobian_2(*up,*zp,*dzp,*rp,true,*outStream);
      *outStream << "\n\nCheck Full Hessian of PDE Constraint\n";
      con->checkApplyAdjointHessian(x,*pp,d,x,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of PDE Constraint\n";
      con->checkApplyAdjointHessian_11(*up,*zp,*pp,*dup,*dualup,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of PDE Constraint\n";
      con->checkApplyAdjointHessian_21(*up,*zp,*pp,*dzp,*dualup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of PDE Constraint\n";
      con->checkApplyAdjointHessian_12(*up,*zp,*pp,*dup,*dualzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of PDE Constraint\n";
      con->checkApplyAdjointHessian_22(*up,*zp,*pp,*dzp,*dualzp,true,*outStream);
      *outStream << "\n";
      con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
      *outStream << "\n";
      con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
      *outStream << "\n";
      con->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);

      *outStream << "\n\nCheck Full Jacobian of Filter\n";
      conFilter->checkApplyJacobian(x,d,*rp,true,*outStream);
      *outStream << "\n\nCheck Jacobian_1 of Filter\n";
      conFilter->checkApplyJacobian_1(*up,*zp,*dup,*rp,true,*outStream);
      *outStream << "\n\nCheck Jacobian_2 of Filter\n";
      conFilter->checkApplyJacobian_2(*up,*zp,*dzp,*rp,true,*outStream);
      *outStream << "\n\nCheck Full Hessian of Filter\n";
      conFilter->checkApplyAdjointHessian(x,*pp,d,x,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of Filter\n";
      conFilter->checkApplyAdjointHessian_11(*up,*zp,*pp,*dup,*dualup,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of Filter\n";
      conFilter->checkApplyAdjointHessian_21(*up,*zp,*pp,*dzp,*dualup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of Filter\n";
      conFilter->checkApplyAdjointHessian_12(*up,*zp,*pp,*dup,*dualzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of Filter\n";
      conFilter->checkApplyAdjointHessian_22(*up,*zp,*pp,*dzp,*dualzp,true,*outStream);
      *outStream << "\n";
      conFilter->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
      *outStream << "\n";
      conFilter->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
      *outStream << "\n";
      conFilter->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);

      *outStream << "\n\nCheck Full Jacobian of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyJacobian(x,d,*rp,true,*outStream);
      *outStream << "\n\nCheck Jacobian_1 of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyJacobian_1(*up,*zp,*dup,*rp,true,*outStream);
      *outStream << "\n\nCheck Jacobian_2 of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyJacobian_2(*up,*zp,*dzp,*rp,true,*outStream);
      *outStream << "\n\nCheck Full Hessian of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyAdjointHessian(x,*pp,d,x,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyAdjointHessian_11(*up,*zp,*pp,*dup,*dualup,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyAdjointHessian_21(*up,*zp,*pp,*dzp,*dualup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyAdjointHessian_12(*up,*zp,*pp,*dup,*dualzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of Filtered PDE Constraint\n";
      pdeWithFilter->checkApplyAdjointHessian_22(*up,*zp,*pp,*dzp,*dualzp,true,*outStream);
      *outStream << "\n";
      pdeWithFilter->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
      *outStream << "\n";
      pdeWithFilter->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
      *outStream << "\n";
      pdeWithFilter->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);

      *outStream << "\n\nCheck Gradient of Reduced Objective Function\n";
      objRed->checkGradient(*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian of Reduced Objective Function\n";
      objRed->checkHessVec(*zp,*dzp,true,*outStream);

      *outStream << "\n\nCheck Full Jacobian of Volume Constraint\n";
      vcon->checkApplyJacobian(*zp,*dzp,*c1p,true,*outStream);
      *outStream << "\n";
      vcon->checkAdjointConsistencyJacobian(*c1p,*dzp,*zp,true,*outStream);
      *outStream << "\n\nCheck Full Hessian of Volume Constraint\n";
      vcon->checkApplyAdjointHessian(*zp,*c2p,*dzp,*zp,true,*outStream);

      //*outStream << "\n\nCheck Gradient of Augmented Lagrangian Function\n";
      //augLag.checkGradient(*zp,*dzp,true,*outStream);
      //*outStream << "\n\nCheck Hessian of Augmented Lagrangian Function\n";
      //augLag.checkHessVec(*zp,*dzp,true,*outStream);
      //*outStream << "\n";
    }

    parlist->sublist("Step").set("Type","Augmented Lagrangian");
    ROL::OptimizationSolver<RealT> optSolver(opt, *parlist);
    Teuchos::Time algoTimer("Algorithm Time", true);
    optSolver.solve(*outStream);
    //ROL::Algorithm<RealT> algo("Augmented Lagrangian",*parlist,false);
    //algo.run(*(opt.getSolutionVector()),*c2p,augLag,*(opt.getConstraint()),*(opt.getBoundConstraint()),true,*outStream);
    algoTimer.stop();
    *outStream << "Total optimization time = " << algoTimer.totalElapsedTime() << " seconds.\n";

    // Output.
    pdecon->printMeshData(*outStream);
    con->solve(*rp,*up,*zp,tol);
    pdecon->outputTpetraVector(u_rcp,"state.txt");
    pdecon->outputTpetraVector(z_rcp,"density.txt");
    //Teuchos::Array<RealT> res(1,0);
    //con->value(*rp,*up,*zp,tol);
    //r_rcp->norm2(res.view(0,1));
    //*outStream << "Residual Norm: " << res[0] << std::endl;
    //errorFlag += (res[0] > 1.e-6 ? 1 : 0);
    //pdecon->outputTpetraData();

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
