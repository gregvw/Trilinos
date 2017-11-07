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
    \brief Shows how to solve the obstacle problem.
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
#include "ROL_Bounds.hpp"
#include "ROL_Bundle.hpp"

#include "../TOOLS/meshmanager.hpp"
#include "../TOOLS/pdeobjective.hpp"
#include "../TOOLS/pdevector.hpp"
#include "pde_obstacle.hpp"
#include "energy_objective.hpp"

typedef double RealT;


template <class Real>
Real evaluateUpperBound(const std::vector<Real> & coord) {
  Real distance(0), one(1);
  if (coord[1] < coord[0]) {
    if (coord[1] < one - coord[0]) {
      distance = coord[1];
    }
    else {
      distance = one - coord[0];
    }
  }
  else {
    if (coord[1] < one - coord[0]) {
      distance = coord[0];
    }
    else {
      distance = 1 - coord[1];
    }
  }
  return distance;
}


template <class Real>
void computeUpperBound(const std::shared_ptr<Tpetra::MultiVector<> > & ubVec,
                       const std::shared_ptr<const FE<Real> > & fe,
                       const std::shared_ptr<Intrepid::FieldContainer<Real> > & cellNodes,
                       const std::shared_ptr<Intrepid::FieldContainer<int> > & cellDofs,
                       const Teuchos::Array<int> & cellIds) {
  int c = fe->gradN()->dimension(0);
  int f = fe->gradN()->dimension(1);
  int d = fe->gradN()->dimension(3);
  std::shared_ptr<Intrepid::FieldContainer<Real> > dofPoints =
    std::make_shared<Intrepid::FieldContainer<Real>>(c,f,d);
  fe->computeDofCoords(dofPoints, cellNodes);
  
  std::vector<Real> coord(d);
  for (int i = 0; i < c; ++i) {
    for (int j = 0; j < f; ++j) {
      int fidx = (*cellDofs)(cellIds[i],j);
      if (ubVec->getMap()->isNodeGlobalElement(fidx)) {
        for (int k = 0; k < d; ++k) {
          coord[k] = (*dofPoints)(i,j,k);
        }
        ubVec->replaceGlobalValue(fidx,
                                  0,
                                  evaluateUpperBound<Real>(coord));
      }
    }
  }
} 


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

    /*** Initialize PDE describing the obstacle problem ***/
    std::shared_ptr<MeshManager<RealT> > meshMgr
      = std::make_shared<MeshManager_Rectangle<RealT>>(*parlist);
    std::shared_ptr<PDE_Obstacle<RealT> > pde
      = std::make_shared<PDE_Obstacle<RealT>>(*parlist);
    std::shared_ptr<EnergyObjective<RealT> > obj
      = std::make_shared<EnergyObjective<RealT>>(pde,meshMgr,comm,*parlist,*outStream);
    std::shared_ptr<Assembler<RealT> > assembler = obj->getAssembler();

    // Create state vector and set to zeroes
    std::shared_ptr<Tpetra::MultiVector<> > u_rcp = assembler->createStateVector();
    u_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > up
      = std::make_shared<PDE_PrimalSimVector<RealT>>(u_rcp,pde,assembler);
    // Create state direction vector and set to random
    std::shared_ptr<Tpetra::MultiVector<> > du_rcp = assembler->createStateVector();
    du_rcp->randomize();
    std::shared_ptr<ROL::Vector<RealT> > dup
      = std::make_shared<PDE_PrimalSimVector<RealT>>(du_rcp,pde,assembler);

    // Build bound constraints
    std::shared_ptr<Tpetra::MultiVector<> > lo_rcp = assembler->createStateVector();
    std::shared_ptr<Tpetra::MultiVector<> > hi_rcp = assembler->createStateVector();
    lo_rcp->putScalar(0.0); hi_rcp->putScalar(1.0);
    computeUpperBound<RealT>(hi_rcp,pde->getFE(),
                             pde->getCellNodes(),
                             assembler->getDofManager()->getCellDofs(),
                             assembler->getCellIds());
    std::shared_ptr<ROL::Vector<RealT> > lop
      = std::make_shared<PDE_PrimalSimVector<RealT>>(lo_rcp,pde,assembler);
    std::shared_ptr<ROL::Vector<RealT> > hip
      = std::make_shared<PDE_PrimalSimVector<RealT>>(hi_rcp,pde,assembler);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd
      = std::make_shared<ROL::Bounds<RealT>>(lop,hip);

    // Run derivative checks
    obj->checkGradient(*up,*dup,true,*outStream);
    obj->checkHessVec(*up,*dup,true,*outStream);

    du_rcp->putScalar(0.4);
    up->set(*dup);
    ROL::Algorithm<RealT> algoTR("Trust Region",*parlist,false);
    std::clock_t timerTR = std::clock();
    algoTR.run(*up,*obj,*bnd,true,*outStream);
    *outStream << "Trust Region Time: "
               << static_cast<RealT>(std::clock()-timerTR)/static_cast<RealT>(CLOCKS_PER_SEC)
               << " seconds." << std::endl << std::endl;

    up->set(*dup);
    ROL::Algorithm<RealT> algoPDAS("Primal Dual Active Set",*parlist,false);
    std::clock_t timerPDAS = std::clock();
    algoPDAS.run(*up,*obj,*bnd,true,*outStream);
    *outStream << "PD Active Set Time: "
               << static_cast<RealT>(std::clock()-timerPDAS)/static_cast<RealT>(CLOCKS_PER_SEC)
               << " seconds." << std::endl << std::endl;

    // Output.
    assembler->printMeshData(*outStream);
    assembler->outputTpetraVector(u_rcp,"state.txt");
    assembler->outputTpetraVector(hi_rcp,"upperBound.txt");

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
