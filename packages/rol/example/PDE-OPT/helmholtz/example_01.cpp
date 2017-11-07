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
    \brief Shows how to solve the optimal control of Helmholtz problem.
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
#include "ROL_Reduced_Objective_SimOpt.hpp"

#include "../TOOLS/linearpdeconstraint.hpp"
#include "../TOOLS/pdeobjective.hpp"
#include "../TOOLS/pdevector.hpp"
#include "../TOOLS/meshmanager.hpp"

#include "pde_helmholtz.hpp"
#include "obj_helmholtz.hpp"

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
    RealT tol(1e-8);// one(1);

    /*** Read in XML input ***/
    std::string filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );

    /*** Initialize main data structure. ***/
    std::shared_ptr<MeshManager<RealT> > meshMgr
      = std::make_shared<MeshManager_Rectangle<RealT>>(*parlist);
    // Initialize PDE describing elasticity equations.
    std::shared_ptr<PDE_Helmholtz<RealT> > pde
      = std::make_shared<PDE_Helmholtz<RealT>>(*parlist);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
      = std::make_shared<Linear_PDE_Constraint<RealT>>(pde,meshMgr,comm,*parlist,*outStream);
    // Cast the constraint and get the assembler.
    std::shared_ptr<Linear_PDE_Constraint<RealT> > pdecon
      = std::dynamic_pointer_cast<Linear_PDE_Constraint<RealT> >(con);
    std::shared_ptr<Assembler<RealT> > assembler = pdecon->getAssembler();
    con->setSolveParameters(*parlist);

    // Create state vector.
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp = assembler->createStateVector();
    u_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > up
      = std::make_shared<PDE_PrimalSimVector<RealT>>(u_rcp,pde,assembler,*parlist);
    std::shared_ptr<Tpetra::MultiVector<> > p_rcp = assembler->createStateVector();
    p_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > pp
      = std::make_shared<PDE_PrimalSimVector<RealT>>(p_rcp,pde,assembler,*parlist);
    // Create control vector.
    std::shared_ptr<Tpetra::MultiVector<> > z_rcp = assembler->createControlVector();
    z_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > zp
      = std::make_shared<PDE_PrimalOptVector<RealT>>(z_rcp,pde,assembler,*parlist);
    // Create residual vector.
    std::shared_ptr<Tpetra::MultiVector<> > r_rcp = assembler->createResidualVector();
    r_rcp->putScalar(0.0);
    std::shared_ptr<ROL::Vector<RealT> > rp
      = std::make_shared<PDE_DualSimVector<RealT>>(r_rcp,pde,assembler,*parlist);
    // Create state direction vector.
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp = assembler->createStateVector();
    du_rcp->randomize();
    //du_rcp->putScalar(0);
    std::shared_ptr<ROL::Vector<RealT> > dup
      = std::make_shared<PDE_PrimalSimVector<RealT>>(du_rcp,pde,assembler,*parlist);
    // Create control direction vector.
    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp = assembler->createControlVector();
    dz_rcp->randomize();
    //dz_rcp->putScalar(0);
    std::shared_ptr<ROL::Vector<RealT> > dzp
      = std::make_shared<PDE_PrimalOptVector<RealT>>(dz_rcp,pde,assembler,*parlist);
    // Create control test vector.
    std::shared_ptr<Tpetra::MultiVector<> > rz_rcp = assembler->createControlVector();
    rz_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > rzp
      = std::make_shared<PDE_PrimalOptVector<RealT>>(rz_rcp,pde,assembler,*parlist);

    std::shared_ptr<Tpetra::MultiVector<> > dualu_rcp = assembler->createStateVector();
    std::shared_ptr<ROL::Vector<RealT> > dualup
      = std::make_shared<PDE_DualSimVector<RealT>>(dualu_rcp,pde,assembler,*parlist);
    std::shared_ptr<Tpetra::MultiVector<> > dualz_rcp = assembler->createControlVector();
    std::shared_ptr<ROL::Vector<RealT> > dualzp
      = std::make_shared<PDE_DualOptVector<RealT>>(dualz_rcp,pde,assembler,*parlist);

    // Create ROL SimOpt vectors.
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    // Initialize compliance objective function.
    bool storage = parlist->sublist("Problem").get("Use Storage",true);
    std::vector<std::shared_ptr<QoI<RealT> > > qoi_vec(2,nullptr);
    qoi_vec[0] = std::make_shared<QoI_Helmholtz_StateTracking<RealT>(pde->getFE(>(),
                                                                     pde->getFieldHelper(),
                                                                     *parlist));
    qoi_vec[1] = std::make_shared<QoI_Helmholtz_ControlPenalty<RealT>(pde->getFE(>(),
                                                                      pde->getFieldHelper(),
                                                                      *parlist));
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj
      = std::make_shared<PDE_Objective<RealT>>(qoi_vec,assembler);
    std::shared_ptr<ROL::Reduced_Objective_SimOpt<RealT> > robj
      = std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, zp, pp, storage, false);

    // Run derivative checks
    bool checkDeriv = parlist->sublist("Problem").get("Check derivatives",false);
    if ( checkDeriv ) {
      *outStream << "\n\nCheck Opt Vector\n";
      zp->checkVector(*dzp,*rzp,true,*outStream);

      std::vector<std::shared_ptr<ROL::Objective_SimOpt<RealT> > > obj_vec(2,nullptr);
      obj_vec[0] = std::make_shared<IntegralObjective<RealT>>(qoi_vec[0],assembler);
      obj_vec[1] = std::make_shared<IntegralObjective<RealT>>(qoi_vec[1],assembler);

      *outStream << "\n\nCheck Gradient of State Objective Function\n";
      obj_vec[0]->checkGradient(x,d,true,*outStream);
      *outStream << "\n\nCheck Gradient_1 of State Objective Function\n";
      obj_vec[0]->checkGradient_1(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Gradient_2 of State Objective Function\n";
      obj_vec[0]->checkGradient_2(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian of State Objective Function\n";
      obj_vec[0]->checkHessVec(x,d,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of State Objective Function\n";
      obj_vec[0]->checkHessVec_11(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of State Objective Function\n";
      obj_vec[0]->checkHessVec_12(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of State Objective Function\n";
      obj_vec[0]->checkHessVec_21(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of State Objective Function\n";
      obj_vec[0]->checkHessVec_22(*up,*zp,*dzp,true,*outStream);

      *outStream << "\n\nCheck Gradient of Control Objective Function\n";
      obj_vec[1]->checkGradient(x,d,true,*outStream);
      *outStream << "\n\nCheck Gradient_1 of Control Objective Function\n";
      obj_vec[1]->checkGradient_1(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Gradient_2 of Control Objective Function\n";
      obj_vec[1]->checkGradient_2(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian of Control Objective Function\n";
      obj_vec[1]->checkHessVec(x,d,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of Control Objective Function\n";
      obj_vec[1]->checkHessVec_11(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of State Objective Function\n";
      obj_vec[1]->checkHessVec_12(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of State Objective Function\n";
      obj_vec[1]->checkHessVec_21(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of Control Objective Function\n";
      obj_vec[1]->checkHessVec_22(*up,*zp,*dzp,true,*outStream);

      *outStream << "\n\nCheck Gradient of Full Objective Function\n";
      obj->checkGradient(x,d,true,*outStream);
      *outStream << "\n\nCheck Gradient_1 of Full Objective Function\n";
      obj->checkGradient_1(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Gradient_2 of Full Objective Function\n";
      obj->checkGradient_2(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian of Full Objective Function\n";
      obj->checkHessVec(x,d,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of Full Objective Function\n";
      obj->checkHessVec_11(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of State Objective Function\n";
      obj->checkHessVec_12(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of State Objective Function\n";
      obj->checkHessVec_21(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of Full Objective Function\n";
      obj->checkHessVec_22(*up,*zp,*dzp,true,*outStream);

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

      *outStream << "\n\nCheck Gradient of Reduced Objective Function\n";
      robj->checkGradient(*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian of Reduced Objective Function\n";
      robj->checkHessVec(*zp,*dzp,true,*outStream);
    }

    // Output uncontrolled state.
    zp->zero();
    pdecon->printMeshData(*outStream);
    con->solve(*rp,*up,*zp,tol);
    pdecon->outputTpetraVector(u_rcp,"state_uncontrolled.txt");

    ROL::Algorithm<RealT> algo("Trust Region",*parlist,false);
    Teuchos::Time algoTimer("Algorithm Time", true);
    algo.run(*zp,*robj,true,*outStream);
    algoTimer.stop();
    *outStream << "Total optimization time = " << algoTimer.totalElapsedTime() << " seconds.\n";

    // Output.
    pdecon->printMeshData(*outStream);
    con->solve(*rp,*up,*zp,tol);
    pdecon->outputTpetraVector(u_rcp,"state.txt");
    pdecon->outputTpetraVector(z_rcp,"control.txt");

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
