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

/*! \file  example_07.cpp
    \brief Shows how to solve a steady Burgers' optimal control problem using
           full-space methods.
*/

#include "ROL_Algorithm.hpp"

#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_BPOEObjective.hpp"
#include "ROL_RiskBoundConstraint.hpp"
#include "ROL_RiskVector.hpp"

#include "ROL_MonteCarloGenerator.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>

#include "example_07.hpp"

typedef double RealT;
typedef H1VectorPrimal<RealT> PrimalStateVector;
typedef H1VectorDual<RealT> DualStateVector;
typedef L2VectorPrimal<RealT> PrimalControlVector;
typedef L2VectorDual<RealT> DualControlVector;
typedef H1VectorDual<RealT> PrimalConstraintVector;
typedef H1VectorPrimal<RealT> DualConstraintVector;

int main(int argc, char *argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  std::shared_ptr<const Teuchos::Comm<int> > comm
    = Teuchos::DefaultComm<int>::getComm();

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  bool print = (iprint>0);
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (print)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  bool print0 = print && !(comm->getRank());
  std::shared_ptr<std::ostream> outStream0;
  if (print0)
    outStream0 = &std::cout, false;
  else
    outStream0 = &bhs, false;

  int errorFlag  = 0;

  // *** Example body.

  try {
    /*************************************************************************/
    /************* INITIALIZE BURGERS FEM CLASS ******************************/
    /*************************************************************************/
    int nx    = 512;   // Set spatial discretization.
    RealT x   = 0.0;   // Set penalty parameter.
    RealT nl  = 1.0;   // Nonlinearity parameter (1 = Burgers, 0 = linear).
    RealT cH1 = 1.0;   // Scale for derivative term in H1 norm.
    RealT cL2 = 0.0;   // Scale for mass term in H1 norm.
    std::shared_ptr<BurgersFEM<RealT> > fem
      = std::make_shared<BurgersFEM<RealT>>(nx,nl,cH1,cL2);
    fem->test_inverse_mass(*outStream0);
    fem->test_inverse_H1(*outStream0);
    /*************************************************************************/
    /************* INITIALIZE SIMOPT OBJECTIVE FUNCTION **********************/
    /*************************************************************************/
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > pobj
      = std::make_shared<Objective_BurgersControl<RealT>>(fem,x);
    /*************************************************************************/
    /************* INITIALIZE SIMOPT EQUALITY CONSTRAINT *********************/
    /*************************************************************************/
    bool hess = true;
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > pcon
      = std::make_shared<Constraint_BurgersControl<RealT>>(fem,hess);
    /*************************************************************************/
    /************* INITIALIZE VECTOR STORAGE *********************************/
    /*************************************************************************/
    // INITIALIZE CONTROL VECTORS
    std::shared_ptr<std::vector<RealT> > z_rcp
      = std::make_shared<std::vector<RealT>>(nx+2, 1.0);
    std::shared_ptr<std::vector<RealT> > gz_rcp
      = std::make_shared<std::vector<RealT>>(nx+2, 1.0);
    std::shared_ptr<std::vector<RealT> > yz_rcp
      = std::make_shared<std::vector<RealT>>(nx+2, 1.0);
    for (int i=0; i<nx+2; i++) {
      (*yz_rcp)[i] = 2.0*random<RealT>(comm)-1.0;
    }
    std::shared_ptr<ROL::Vector<RealT> > zp
      = std::make_shared<PrimalControlVector>(z_rcp,fem);
    std::shared_ptr<ROL::Vector<RealT> > gzp
      = std::make_shared<DualControlVector>(gz_rcp,fem);
    std::shared_ptr<ROL::Vector<RealT> > yzp
      = std::make_shared<PrimalControlVector>(yz_rcp,fem);
    RealT zvar = random<RealT>(comm);
    RealT gvar = random<RealT>(comm);
    RealT yvar = random<RealT>(comm);
    std::shared_ptr<Teuchos::ParameterList> bpoelist = std::make_shared<Teuchos::ParameterList>();
    bpoelist->sublist("SOL").sublist("Risk Measure").set("Name","bPOE");
    ROL::RiskVector<RealT> z(bpoelist,zp,zvar), g(bpoelist,gzp,gvar), y(bpoelist,yzp,yvar);
    // INITIALIZE STATE VECTORS
    std::shared_ptr<std::vector<RealT> > u_rcp
      = std::make_shared<std::vector<RealT>>(nx, 1.0);
    std::shared_ptr<std::vector<RealT> > gu_rcp
      = std::make_shared<std::vector<RealT>>(nx, 1.0);
    std::shared_ptr<ROL::Vector<RealT> > up
      = std::make_shared<PrimalStateVector>(u_rcp,fem);
    std::shared_ptr<ROL::Vector<RealT> > gup
      = std::make_shared<DualStateVector>(gu_rcp,fem);
    // INITIALIZE CONSTRAINT VECTORS
    std::shared_ptr<std::vector<RealT> > c_rcp
      = std::make_shared<std::vector<RealT>>(nx, 1.0);
    std::shared_ptr<std::vector<RealT> > l_rcp
      = std::make_shared<std::vector<RealT>>(nx, 1.0);
    for (int i=0; i<nx; i++) {
      (*l_rcp)[i] = random<RealT>(comm);
    }
    std::shared_ptr<ROL::Vector<RealT> > cp
      = std::make_shared<PrimalConstraintVector>(c_rcp,fem);
    std::shared_ptr<ROL::Vector<RealT> > lp
      = std::make_shared<DualConstraintVector>(l_rcp,fem);
    /*************************************************************************/
    /************* INITIALIZE SAMPLE GENERATOR *******************************/
    /*************************************************************************/
    int dim = 4, nSamp = 1000;
    std::vector<RealT> tmp(2,0.0); tmp[0] = -1.0; tmp[1] = 1.0;
    std::vector<std::vector<RealT> > bounds(dim,tmp);
    std::shared_ptr<ROL::BatchManager<RealT> > bman
      = std::make_shared<L2VectorBatchManager<RealT,int>>(comm);
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler
      = Teuchos::rcp(new ROL::MonteCarloGenerator<RealT>(
          nSamp,bounds,bman,false,false,100));
    /*************************************************************************/
    /************* INITIALIZE RISK-AVERSE OBJECTIVE FUNCTION *****************/
    /*************************************************************************/
    bool storage = true, fdhess = false;
    std::shared_ptr<ROL::Objective<RealT> > robj
      = Teuchos::rcp(new ROL::Reduced_Objective_SimOpt<RealT>(
          pobj,pcon,up,zp,lp,gup,gzp,cp,storage,fdhess));
    RealT order = 2.0, threshold = -0.85*(1.0-x);
    std::shared_ptr<ROL::Objective<RealT> > obj
      = Teuchos::rcp(new ROL::BPOEObjective<RealT>(
          robj,order,threshold,sampler,storage));
    /*************************************************************************/
    /************* INITIALIZE BOUND CONSTRAINTS ******************************/
    /*************************************************************************/
    std::vector<RealT> Zlo(nx+2,0.0), Zhi(nx+2,10.0);
    for (int i = 0; i < nx+2; i++) {
      if ( i < (int)((nx+2)/3) ) {
        Zlo[i] = -1.0;
        Zhi[i] = 1.0;
      }
      if ( i >= (int)((nx+2)/3) && i < (int)(2*(nx+2)/3) ) {
        Zlo[i] = 1.0;
        Zhi[i] = 5.0;
      }
      if ( i >= (int)(2*(nx+2)/3) ) {
        Zlo[i] = 5.0;
        Zhi[i] = 10.0;
      }
    }
    std::shared_ptr<ROL::BoundConstraint<RealT> > Zbnd
      = std::make_shared<L2BoundConstraint<RealT>>(Zlo,Zhi,fem);
    std::shared_ptr<ROL::BoundConstraint<RealT> > bnd
      = std::make_shared<ROL::RiskBoundConstraint<RealT>>(bpoelist,Zbnd);
    /*************************************************************************/
    /************* CHECK DERIVATIVES AND CONSISTENCY *************************/
    /*************************************************************************/
    // CHECK OBJECTIVE DERIVATIVES
    bool derivcheck = false;
    if (derivcheck) {
      int nranks = sampler->numBatches();
      for (int pid = 0; pid < nranks; pid++) {
        if ( pid == sampler->batchID() ) {
          for (int i = sampler->start(); i < sampler->numMySamples(); i++) {
            *outStream << "Sample " << i << "  Rank " << sampler->batchID() << "\n";
            *outStream << "(" << sampler->getMyPoint(i)[0] << ", "
                              << sampler->getMyPoint(i)[1] << ", "
                              << sampler->getMyPoint(i)[2] << ", "
                              << sampler->getMyPoint(i)[3] << ")\n";
            pcon->setParameter(sampler->getMyPoint(i));
            pcon->checkSolve(*up,*zp,*cp,print,*outStream);
            robj->setParameter(sampler->getMyPoint(i));
            *outStream << "\n";
            robj->checkGradient(*zp,*gzp,*yzp,print,*outStream);
            robj->checkHessVec(*zp,*gzp,*yzp,print,*outStream);
            *outStream << "\n\n";
          }
        }
        comm->barrier();
      }
    }
    obj->checkGradient(z,g,y,print0,*outStream0);
    obj->checkHessVec(z,g,y,print0,*outStream0);
    /*************************************************************************/
    /************* RUN OPTIMIZATION ******************************************/
    /*************************************************************************/
    // READ IN XML INPUT
    std::string filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist
      = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    // RUN OPTIMIZATION
    ROL::Algorithm<RealT> algo("Trust Region",*parlist,false);
    zp->zero();
    algo.run(z, g, *obj, *bnd, print0, *outStream0);
    /*************************************************************************/
    /************* PRINT CONTROL AND STATE TO SCREEN *************************/
    /*************************************************************************/
    if ( print0 ) {
      std::ofstream ofs;
      ofs.open("output_example_09.txt",std::ofstream::out);
      for ( int i = 0; i < nx+2; i++ ) {
        ofs << std::scientific << std::setprecision(10);
        ofs << std::setw(20) << std::left << (RealT)i/((RealT)nx+1.0);
        ofs << std::setw(20) << std::left << (*z_rcp)[i];
        ofs << "\n";
      }
      ofs.close();
    }
    *outStream0 << "Scalar Parameter: " << z.getStatistic(0) << "\n\n";
  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  comm->barrier();
  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;
}
