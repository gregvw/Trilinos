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

/*! \file  example_08.cpp
    \brief Shows how to minimize volume subject to a constraint on
           compliance.
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

#include "ROL_Algorithm.hpp"
#include "ROL_AugmentedLagrangian.hpp"
#include "ROL_ScaledStdVector.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_Reduced_Constraint_SimOpt.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_CompositeConstraint_SimOpt.hpp"
#include "ROL_ConstraintFromObjective.hpp"
#include "ROL_OptimizationSolver.hpp"

#include "../../TOOLS/pdeconstraint.hpp"
#include "../../TOOLS/linearpdeconstraint.hpp"
#include "../../TOOLS/pdeobjective.hpp"
#include "../../TOOLS/pdevector.hpp"
#include "../../TOOLS/integralconstraint.hpp"
#include "../../TOOLS/meshreader.hpp"
#include "obj_topo-opt.hpp"
#include "mesh_topo-opt.hpp"
#include "pde_elasticity.hpp"
#include "pde_filter.hpp"

typedef double RealT;

int main(int argc, char *argv[]) {
  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing

  /*** Initialize communicator. ***/
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &bhs);
  std::shared_ptr<const Teuchos::Comm<int> > comm
    = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
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
    std::string filename = "input_ex08.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );

    // Retrieve parameters.
    const RealT cmpFactor    = parlist->sublist("Problem").get("Compliance Factor", 1.1);
    const RealT objFactor    = parlist->sublist("Problem").get("Objective Scaling", 1e-4);

    /*** Initialize main data structure. ***/
    int probDim = parlist->sublist("Problem").get("Problem Dimension",2);
    std::shared_ptr<MeshManager<RealT> > meshMgr;
    if (probDim == 2) {
      meshMgr = std::make_shared<MeshManager_TopoOpt<RealT>>(*parlist);
    } else if (probDim == 3) {
      meshMgr = std::make_shared<MeshReader<RealT>>(*parlist);
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
        ">>> PDE-OPT/topo-opt/elasticity/example_08.cpp: Problem dim is not 2 or 3!");
    }
    // Initialize PDE describing elasticity equations.
    std::shared_ptr<PDE_Elasticity<RealT> > pde
      = std::make_shared<PDE_Elasticity<RealT>>(*parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
      = std::make_shared<PDE_Constraint<RealT>>(pde,meshMgr,comm,*parlist,*outStream);
    // Initialize the filter PDE.
    std::shared_ptr<PDE_Filter<RealT> > pdeFilter
      = std::make_shared<PDE_Filter<RealT>>(*parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > conFilter
      = std::make_shared<Linear_PDE_Constraint<RealT>>(pdeFilter,meshMgr,comm,*parlist,*outStream);
    // Cast the constraint and get the assembler.
    std::shared_ptr<PDE_Constraint<RealT> > pdecon
      = std::dynamic_pointer_cast<PDE_Constraint<RealT> >(con);
    std::shared_ptr<Assembler<RealT> > assembler = pdecon->getAssembler();
    con->setSolveParameters(*parlist);

    // Create state vector.
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp, p_rcp, z_rcp, r_rcp;
    u_rcp = assembler->createStateVector();    u_rcp->putScalar(0.0);
    p_rcp = assembler->createStateVector();    p_rcp->putScalar(0.0);
    z_rcp = assembler->createControlVector();  z_rcp->putScalar(1.0);
    r_rcp = assembler->createResidualVector(); r_rcp->putScalar(0.0);
    std::shared_ptr<ROL::Vector<RealT> > up, pp, zp, rp;
    up = std::make_shared<PDE_PrimalSimVector<RealT>>(u_rcp,pde,assembler,*parlist);
    pp = std::make_shared<PDE_PrimalSimVector<RealT>>(p_rcp,pde,assembler,*parlist);
    zp = std::make_shared<PDE_PrimalOptVector<RealT>>(z_rcp,pde,assembler,*parlist);
    rp = std::make_shared<PDE_DualSimVector<RealT>>(r_rcp,pde,assembler,*parlist);

    // Initialize "filtered" of "unfiltered" constraint.
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > pdeWithFilter;
    bool useFilter = parlist->sublist("Problem").get("Use Filter", true);
    if (useFilter) {
      bool useStorage = parlist->sublist("Problem").get("Use State Storage",true);
      pdeWithFilter
        = Teuchos::rcp(new ROL::CompositeConstraint_SimOpt<RealT>(con, conFilter,
                       *rp, *rp, *up, *zp, *zp, useStorage));
    }
    else {
      pdeWithFilter = con;
    }
    pdeWithFilter->setSolveParameters(*parlist);

    // Initialize volume objective.
    std::shared_ptr<QoI<RealT> > qoi_vol
      = std::make_shared<QoI_VolumeObj_TopoOpt<RealT>(pde->getFE(),pde->getFieldHelper>());
    std::shared_ptr<ROL::Objective<RealT> > vobj
      = std::make_shared<IntegralOptObjective<RealT>>(qoi_vol,assembler);

    // Initialize compliance inequality constraint.
    con->value(*rp, *up, *zp, tol);
    RealT objScaling = objFactor, rnorm2 = rp->dot(*rp);
    if (rnorm2 > 1e2*ROL::ROL_EPSILON<RealT>()) {
      objScaling /= rnorm2;
    }
    std::vector<std::shared_ptr<QoI<RealT> > > qoi_cmp(1,nullptr);
    qoi_cmp[0]
      = std::make_shared<QoI_TopoOpt<RealT>(pde->getFE>(),
                                            pde->getFieldHelper(), objScaling));
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > cobj
      = std::make_shared<PDE_Objective<RealT>>(qoi_cmp,assembler);

    // Initialize reduced compliance objective function.
    bool storage = parlist->sublist("Problem").get("Use state storage",true);
    std::shared_ptr<ROL::SimController<RealT> > stateStore
      = std::make_shared<ROL::SimController<RealT>>();
    std::shared_ptr<ROL::Reduced_Objective_SimOpt<RealT> > cobjRed
      = Teuchos::rcp(new  ROL::Reduced_Objective_SimOpt<RealT>(cobj,
                     pdeWithFilter,stateStore,up,zp,pp,storage));

    // Create compliance constraint, multiplier and bounds
    RealT comp = cobjRed->value(*zp,tol);
    std::shared_ptr<ROL::Constraint<RealT> > icon
      = std::make_shared<ROL::ConstraintFromObjective<RealT>>(cobjRed);
    std::shared_ptr<std::vector<RealT> > imul_rcp, iup_rcp;
    std::shared_ptr<ROL::Vector<RealT> > imul, iup;
    imul = std::make_shared<ROL::SingletonVector<RealT>>(0);
    iup  = std::make_shared<ROL::SingletonVector<RealT>>(cmpFactor*comp);
    std::shared_ptr<ROL::BoundConstraint<RealT> > ibnd
      = std::make_shared<ROL::Bounds<RealT>>(*iup,false);

    // Initialize bound constraints.
    std::shared_ptr<Tpetra::MultiVector<> > lo_rcp, hi_rcp;
    lo_rcp = assembler->createControlVector(); lo_rcp->putScalar(0.0);
    hi_rcp = assembler->createControlVector(); hi_rcp->putScalar(1.0);
    std::shared_ptr<ROL::Vector<RealT> > lop, hip;
    lop = std::make_shared<PDE_PrimalOptVector<RealT>>(lo_rcp,pde,assembler);
    hip = std::make_shared<PDE_PrimalOptVector<RealT>>(hi_rcp,pde,assembler);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd
      = std::make_shared<ROL::Bounds<RealT>>(lop,hip);

    // Build optimization problem.
    ROL::OptimizationProblem<RealT> optProb(vobj,zp,bnd,icon,imul,ibnd);

    // Run derivative checks
    bool checkDeriv = parlist->sublist("Problem").get("Check derivatives",false);
    if ( checkDeriv ) {
      optProb.check(*outStream);
    }

    // Build optimization solver and solve.
    ROL::OptimizationSolver<RealT> optSolver(optProb,*parlist);
    Teuchos::Time algoTimer("Algorithm Time", true);
    optSolver.solve(*outStream);
    algoTimer.stop();
    *outStream << "Total optimization time = " << algoTimer.totalElapsedTime() << " seconds.\n";

    // Output.
    pdecon->printMeshData(*outStream);
    con->solve(*rp,*up,*zp,tol);
    pdecon->outputTpetraVector(u_rcp,"state.txt");
    pdecon->outputTpetraVector(z_rcp,"density.txt");

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
