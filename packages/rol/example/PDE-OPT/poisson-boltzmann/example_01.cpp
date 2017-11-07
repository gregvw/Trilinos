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
    \brief Shows how to solve the Poisson-Boltzmann problem.
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

#include "../TOOLS/meshmanager.hpp"
#include "../TOOLS/pdeconstraint.hpp"
#include "../TOOLS/pdeobjective.hpp"
#include "../TOOLS/pdevector.hpp"
#include "pde_poisson_boltzmann.hpp"
#include "obj_poisson_boltzmann.hpp"

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

    /*** Initialize main data structure. ***/
    std::shared_ptr<MeshManager<RealT> > meshMgr
      = std::make_shared<MeshManager_Rectangle<RealT>>(*parlist);
    // Initialize PDE describe Poisson's equation
    std::shared_ptr<PDE_Poisson_Boltzmann<RealT> > pde
      = std::make_shared<PDE_Poisson_Boltzmann<RealT>>(*parlist);
    std::shared_ptr<PDE_Constraint<RealT> > con
      = std::make_shared<PDE_Constraint<RealT>>(pde,meshMgr,comm,*parlist,*outStream);
    // Initialize quadratic objective function
    std::vector<std::shared_ptr<QoI<RealT> > > qoi_vec(2,nullptr);
    qoi_vec[0] = std::make_shared<QoI_L2Tracking_Poisson_Boltzmann<RealT>(pde->getFE>());
    qoi_vec[1] = std::make_shared<QoI_L2Penalty_Poisson_Boltzmann<RealT>(pde->getFE>());
    std::shared_ptr<StdObjective_Poisson_Boltzmann<RealT> > std_obj
      = std::make_shared<StdObjective_Poisson_Boltzmann<RealT>>(*parlist);
    std::shared_ptr<PDE_Objective<RealT> > obj
      = std::make_shared<PDE_Objective<RealT>(qoi_vec,std_obj,con->getAssembler>());

    // Create state vector and set to zeroes
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp = con->getAssembler()->createStateVector();
    u_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > up
      = std::make_shared<PDE_PrimalSimVector<RealT>(u_rcp,pde,con->getAssembler>());
    // Create control vector and set to ones
    std::shared_ptr<Tpetra::MultiVector<> > z_rcp = con->getAssembler()->createControlVector();
    z_rcp->putScalar(1.0);
    std::shared_ptr<ROL::Vector<RealT> > zp
      = std::make_shared<PDE_PrimalOptVector<RealT>(z_rcp,pde,con->getAssembler>());
    // Create residual vector and set to zeros
    std::shared_ptr<Tpetra::MultiVector<> > r_rcp = con->getAssembler()->createResidualVector();
    r_rcp->putScalar(0.0);
    std::shared_ptr<ROL::Vector<RealT> > rp
      = std::make_shared<PDE_DualSimVector<RealT>(r_rcp,pde,con->getAssembler>());
    // Create state direction vector and set to random
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp = con->getAssembler()->createStateVector();
    du_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > dup
      = std::make_shared<PDE_PrimalSimVector<RealT>(du_rcp,pde,con->getAssembler>());
    // Create control direction vector and set to random
    std::shared_ptr<Tpetra::MultiVector<> > dz_rcp = con->getAssembler()->createControlVector();
    //dz_rcp->randomize();
    dz_rcp->putScalar(0.0);
    std::shared_ptr<ROL::Vector<RealT> > dzp
      = std::make_shared<PDE_PrimalOptVector<RealT>(dz_rcp,pde,con->getAssembler>());
    // Create ROL SimOpt vectors
    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> d(dup,dzp);

    // Run derivative checks
    obj->checkGradient(x,d,true,*outStream);
    obj->checkHessVec(x,d,true,*outStream);
    con->checkApplyJacobian(x,d,*up,true,*outStream);
    con->checkApplyAdjointHessian(x,*dup,d,x,true,*outStream);
    con->checkAdjointConsistencyJacobian(*dup,d,x,true,*outStream);
    con->checkInverseJacobian_1(*up,*up,*up,*zp,true,*outStream);
    con->checkInverseAdjointJacobian_1(*up,*up,*up,*zp,true,*outStream);

    ROL::Algorithm<RealT> algo("Composite Step",*parlist,false);
    algo.run(x,*rp,*obj,*con,true,*outStream);

    // Output.
    con->getAssembler()->printMeshData(*outStream);
    RealT tol(1.e-8);
    con->solve(*rp,*up,*zp,tol);
    con->outputTpetraVector(u_rcp,"state.txt");
    con->outputTpetraVector(z_rcp,"control.txt");

    Teuchos::Array<RealT> res(1,0);
    con->value(*rp,*up,*zp,tol);
    r_rcp->norm2(res.view(0,1));
    *outStream << "Residual Norm: " << res[0] << std::endl;
    errorFlag += (res[0] > 1.e-6 ? 1 : 0);
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
