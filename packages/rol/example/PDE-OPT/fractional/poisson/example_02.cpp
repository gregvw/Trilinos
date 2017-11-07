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
    \brief Shows how to solve the Poisson control problem.
*/

#include "Teuchos_Comm.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Version.hpp"

#include <iostream>
#include <algorithm>

#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"

#include "../../TOOLS/meshmanager.hpp"
#include "../../TOOLS/linearpdeconstraint.hpp"
#include "../../TOOLS/pdeobjective.hpp"
#include "../../TOOLS/pdevector.hpp"
#include "pde_fractional_poisson.hpp"
#include "obj_fractional_poisson.hpp"
#include "fractional_constraint.hpp"
#include "fractional_objective.hpp"

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

    /*** Read in XML input ***/
    std::string filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );

    // Set parameters for cylidner mesh
    RealT s     = parlist->sublist("Problem").get("Fractional Power",0.5);
    RealT gamma = (s == 0.5) ? static_cast<RealT>(1)
                  : static_cast<RealT>(3)/(static_cast<RealT>(2)*s) + static_cast<RealT>(1e-3);
    int NX      = parlist->sublist("Geometry").get("NX",20);
    int NY      = parlist->sublist("Geometry").get("NY",20);
    RealT NT    = static_cast<RealT>(NX*NY);
    RealT width = static_cast<RealT>(1) + std::log10(NT)/static_cast<RealT>(3);
    int NI      = static_cast<int>(width * std::sqrt(NT)); 
    parlist->sublist("Geometry").sublist("Cylinder").set("Grading Parameter", gamma);
    parlist->sublist("Geometry").sublist("Cylinder").set("NI",                NI);
    parlist->sublist("Geometry").sublist("Cylinder").set("Height",            width);

    *outStream << std::endl;
    *outStream << "Fractional Power:       " << s     << std::endl;
    *outStream << "Mesh Grading Parameter: " << gamma << std::endl;
    *outStream << "Cylinder Height:        " << width << std::endl;
    *outStream << "Number of Intervals:    " << NI    << std::endl;
    *outStream << std::endl;

    // Initialize 2D Poisson's equation
    std::shared_ptr<MeshManager<RealT> > meshMgr_local
      = std::make_shared<MeshManager_Rectangle<RealT>>(*parlist);
    std::shared_ptr<PDE_Fractional_Poisson_Local<RealT> > pde_local
      = std::make_shared<PDE_Fractional_Poisson_Local<RealT>>(*parlist);
    // Initialize 1D singular Poisson's equation
    std::shared_ptr<MeshManager<RealT> > meshMgr_cylinder
      = std::make_shared<MeshManager_Fractional_Cylinder<RealT>>(*parlist);
    std::shared_ptr<PDE_Fractional_Poisson_Cylinder<RealT> > pde_cylinder
      = std::make_shared<PDE_Fractional_Poisson_Cylinder<RealT>>(*parlist);
    // Build fractional constraint
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > con
      = Teuchos::rcp(new FractionalConstraint<RealT>(pde_local,    meshMgr_local,    comm,
                                                     pde_cylinder, meshMgr_cylinder, comm,
                                                     *parlist, *outStream));
    std::shared_ptr<FractionalConstraint<RealT> > fracCon
      = std::dynamic_pointer_cast<FractionalConstraint<RealT> >(con);
    std::shared_ptr<Assembler<RealT> > assembler = fracCon->getLocalAssembler();

    // Build objective fuction
    std::vector<std::shared_ptr<QoI<RealT> > > qoi_vec(2,nullptr);
    qoi_vec[0] = std::make_shared<QoI_L2Tracking_Fractional_Poisson<RealT>(pde_local->getFE>());
    qoi_vec[1] = std::make_shared<QoI_L2Penalty_Fractional_Poisson<RealT>(pde_local->getFE>());
    RealT stateCost   = parlist->sublist("Problem").get("State Cost",1e0);
    RealT controlCost = parlist->sublist("Problem").get("Control Cost",1e0);
    std::vector<RealT> wts = {stateCost, controlCost};
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > pdeobj
      = std::make_shared<PDE_Objective<RealT>>(qoi_vec,wts,assembler);
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > obj
      = std::make_shared<FractionalObjective<RealT>>(pdeobj);

    // Build vectors
    std::shared_ptr<Tpetra::MultiVector<> > stateVec = assembler->createStateVector();
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp    = std::make_shared<Tpetra::MultiVector<>(stateVec->getMap>(),NI+1);
    std::shared_ptr<Tpetra::MultiVector<> > p_rcp    = std::make_shared<Tpetra::MultiVector<>(stateVec->getMap>(),NI+1);
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp   = std::make_shared<Tpetra::MultiVector<>(stateVec->getMap>(),NI+1);
    u_rcp->randomize();  //u_rcp->putScalar(static_cast<RealT>(1));
    p_rcp->randomize();  //p_rcp->putScalar(static_cast<RealT>(1));
    du_rcp->randomize(); //du_rcp->putScalar(static_cast<RealT>(0));
    std::shared_ptr<ROL::Vector<RealT> > up  = std::make_shared<ROL::TpetraMultiVector<RealT>>(u_rcp);
    std::shared_ptr<ROL::Vector<RealT> > pp  = std::make_shared<ROL::TpetraMultiVector<RealT>>(p_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dup = std::make_shared<ROL::TpetraMultiVector<RealT>>(du_rcp);
    // Create residual vectors
    std::shared_ptr<Tpetra::MultiVector<> > conVec = assembler->createResidualVector();
    std::shared_ptr<Tpetra::MultiVector<> > r_rcp  = std::make_shared<Tpetra::MultiVector<>(conVec->getMap>(),NI+1);
    r_rcp->randomize(); //r_rcp->putScalar(static_cast<RealT>(1));
    std::shared_ptr<ROL::Vector<RealT> > rp = std::make_shared<ROL::TpetraMultiVector<RealT>>(r_rcp);
    // Create control vector and set to ones
    std::shared_ptr<Tpetra::MultiVector<> > z_rcp  = assembler->createControlVector();
    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp = assembler->createControlVector();
    z_rcp->randomize();  //z_rcp->putScalar(static_cast<RealT>(1));
    dz_rcp->randomize(); //dz_rcp->putScalar(static_cast<RealT>(0));
    std::shared_ptr<ROL::Vector<RealT> > zp  = std::make_shared<ROL::TpetraMultiVector<RealT>>(z_rcp);
    std::shared_ptr<ROL::Vector<RealT> > dzp = std::make_shared<ROL::TpetraMultiVector<RealT>>(dz_rcp);
    // Create ROL SimOpt vectors
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    // Build reduced objective function
    bool storage = parlist->sublist("Problem").get("Use State and Adjoint Storage",true);
    std::shared_ptr<ROL::Reduced_Objective_SimOpt<RealT> > objReduced
      = std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(obj, con, up, zp, pp, storage, false);

    // Check derivatives.
    bool checkDeriv = parlist->sublist("Problem").get("Check Derivatives",false);
    if ( checkDeriv ) {
      *outStream << "\n\nCheck Gradient_1 of Full Objective Function\n";
      obj->checkGradient_1(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Gradient_2 of Full Objective Function\n";
      obj->checkGradient_2(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Gradient of Full Objective Function\n";
      obj->checkGradient(x,d,true,*outStream);
      *outStream << "\n\nCheck Hessian_11 of Full Objective Function\n";
      obj->checkHessVec_11(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of Full Objective Function\n";
      obj->checkHessVec_12(*up,*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of Full Objective Function\n";
      obj->checkHessVec_21(*up,*zp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of Full Objective Function\n";
      obj->checkHessVec_22(*up,*zp,*dzp,true,*outStream);
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
      con->checkApplyAdjointHessian_11(*up,*zp,*pp,*dup,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_21 of PDE Constraint\n";
      con->checkApplyAdjointHessian_21(*up,*zp,*pp,*dzp,*dup,true,*outStream);
      *outStream << "\n\nCheck Hessian_12 of PDE Constraint\n";
      con->checkApplyAdjointHessian_12(*up,*zp,*pp,*dup,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian_22 of PDE Constraint\n";
      con->checkApplyAdjointHessian_22(*up,*zp,*pp,*dzp,*dzp,true,*outStream);
      *outStream << "\n";
      con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
      *outStream << "\n";
      con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
      *outStream << "\n";
      *outStream << "\n\nCheck Gradient of Reduced Objective Function\n";
      objReduced->checkGradient(*zp,*dzp,true,*outStream);
      *outStream << "\n\nCheck Hessian of Reduced Objective Function\n";
      objReduced->checkHessVec(*zp,*dzp,true,*outStream);
    }

    // Run optimization
    std::shared_ptr<ROL::Algorithm<RealT> > algo
      = std::make_shared<ROL::Algorithm<RealT>>("Trust Region",*parlist,false);
    zp->zero();
    algo->run(*zp,*objReduced,true,*outStream);

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
