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

#include "test_01.hpp"

typedef double RealT;

template<class Real>
Real random(const ROL::SharedPointer<const Teuchos::Comm<int> > &comm) {
  Real val = 0.0;
  if ( Teuchos::rank<int>(*comm)==0 ) {
    val = (Real)rand()/(Real)RAND_MAX;
  }
  Teuchos::broadcast<int,Real>(*comm,0,1,&val);
  return val;
}

int main(int argc, char* argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  ROL::SharedPointer<const Teuchos::Comm<int> > comm
    = Teuchos::DefaultComm<int>::getComm();

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  ROL::SharedPointer<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0 && Teuchos::rank<int>(*comm)==0)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  int errorFlag  = 0;

  try {
    /**********************************************************************************************/
    /************************* CONSTRUCT ROL ALGORITHM ********************************************/
    /**********************************************************************************************/
    // Get ROL parameterlist
    std::string filename = "input.xml";
    ROL::SharedPointer<Teuchos::ParameterList> parlist = ROL::makeShared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    /**********************************************************************************************/
    /************************* CONSTRUCT VECTORS **************************************************/
    /**********************************************************************************************/
    // Build control vectors
    int nx = 1024;
    ROL::SharedPointer<std::vector<RealT> > x1_rcp  = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> x1(x1_rcp);
    ROL::SharedPointer<std::vector<RealT> > x2_rcp  = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> x2(x2_rcp);
    ROL::SharedPointer<std::vector<RealT> > x3_rcp  = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> x3(x3_rcp);
    ROL::SharedPointer<std::vector<RealT> > z_rcp  = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> z(z_rcp);
    ROL::SharedPointer<std::vector<RealT> > xr_rcp = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> xr(xr_rcp);
    ROL::SharedPointer<std::vector<RealT> > d_rcp  = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> d(d_rcp);
    for ( int i = 0; i < nx+2; i++ ) {
      (*xr_rcp)[i] = random<RealT>(comm);
      (*d_rcp)[i]  = random<RealT>(comm);
    }
    // Build state and adjoint vectors
    ROL::SharedPointer<std::vector<RealT> > u_rcp  = ROL::makeShared<std::vector<RealT>>(nx,1.0);
    ROL::StdVector<RealT> u(u_rcp);
    ROL::SharedPointer<std::vector<RealT> > p_rcp  = ROL::makeShared<std::vector<RealT>>(nx,0.0);
    ROL::StdVector<RealT> p(p_rcp);
    ROL::SharedPointer<ROL::Vector<RealT> > up = &u,false;
    ROL::SharedPointer<ROL::Vector<RealT> > zp = &z,false;
    ROL::SharedPointer<ROL::Vector<RealT> > pp = &p,false;
    /**********************************************************************************************/
    /************************* CONSTRUCT SOL COMPONENTS *******************************************/
    /**********************************************************************************************/
    // Build samplers
    int dim = 4;
    int nSamp = parlist->sublist("Problem Description").get("Number of Samples", 20);
    std::vector<RealT> tmp(2,0.0); tmp[0] = -1.0; tmp[1] = 1.0;
    std::vector<std::vector<RealT> > bounds(dim,tmp);
    ROL::SharedPointer<ROL::BatchManager<RealT> > bman
      = ROL::makeShared<ROL::StdTeuchosBatchManager<RealT,int>>(comm);
    ROL::SharedPointer<ROL::SampleGenerator<RealT> > sampler
      = ROL::makeShared<ROL::MonteCarloGenerator<RealT>>(nSamp,bounds,bman,false,false,100);
    /**********************************************************************************************/
    /************************* CONSTRUCT OBJECTIVE FUNCTION ***************************************/
    /**********************************************************************************************/
    // Build risk-averse objective function
    RealT alpha = 1.e-3;
    ROL::SharedPointer<ROL::Objective_SimOpt<RealT> > pobjSimOpt
      = ROL::makeShared<Objective_BurgersControl<RealT>>(alpha,nx);
    ROL::SharedPointer<ROL::Constraint_SimOpt<RealT> > pconSimOpt
      = ROL::makeShared<Constraint_BurgersControl<RealT>>(nx);
    ROL::SharedPointer<ROL::Objective<RealT> > pObj
      = ROL::makeShared<ROL::Reduced_Objective_SimOpt<RealT>>(pobjSimOpt,pconSimOpt,up,zp,pp);
    ROL::SharedPointer<ROL::Objective<RealT> > obj = ROL::makeShared<ROL::RiskNeutralObjective<RealT>>(pObj, sampler, true);
    // Test parametrized objective functions
    *outStream << "Check Derivatives of Parametrized Objective Function\n";
    x1.set(xr);
    pObj->setParameter(sampler->getMyPoint(0));
    pObj->checkGradient(x1,d,true,*outStream);
    pObj->checkHessVec(x1,d,true,*outStream);
    obj->checkGradient(x1,d,true,*outStream);
    obj->checkHessVec(x1,d,true,*outStream);
    ROL::Algorithm<RealT> algors("Trust Region", *parlist);
    algors.run(z, *obj, true, *outStream);
    /**********************************************************************************************/
    /****************** CONSTRUCT SIMULATED CONSTRAINT AND VECTORS ********************************/
    /**********************************************************************************************/

    // Construct SimulatedConstraint.
    int useW = parlist->sublist("Problem Description").get("Use Constraint Weights", true);
    ROL::SimulatedConstraint<RealT> simcon(sampler, pconSimOpt, useW);
    // Construct SimulatedObjective.
    ROL::SimulatedObjective<RealT> simobj(sampler, pobjSimOpt);
    // Simulated vectors.
    std::vector<ROL::SharedPointer<ROL::Vector<RealT> > > xu_rcp;
    std::vector<ROL::SharedPointer<ROL::Vector<RealT> > > vu_rcp;
    int nvecloc = sampler->numMySamples();
    RealT right = 1, left = 0;
    for( int k=0; k<nvecloc; ++k ) {
      ROL::SharedPointer<std::vector<RealT> > xuk_rcp = ROL::makeShared<std::vector<RealT>>(nx,1.0);
      ROL::SharedPointer<std::vector<RealT> > vuk_rcp = ROL::makeShared<std::vector<RealT>>(nx,1.0);
      ROL::SharedPointer<ROL::Vector<RealT> > xuk = ROL::makeShared<ROL::StdVector<RealT>>( xuk_rcp );
      ROL::SharedPointer<ROL::Vector<RealT> > vuk = ROL::makeShared<ROL::StdVector<RealT>>( vuk_rcp );
      for( int i=0; i<nx; ++i ) {
        (*xuk_rcp)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
        (*vuk_rcp)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      }
      xu_rcp.push_back(xuk);
      vu_rcp.push_back(vuk);
    }
    ROL::SharedPointer<ROL::SimulatedVector<RealT> > xu = ROL::makeShared<ROL::SimulatedVector<RealT>>(xu_rcp, bman);
    ROL::SharedPointer<ROL::SimulatedVector<RealT> > vu = ROL::makeShared<ROL::SimulatedVector<RealT>>(vu_rcp, bman);
    // SimOpt vectors.
    ROL::SharedPointer<std::vector<RealT> > zvec_rcp = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::SharedPointer<ROL::StdVector<RealT> > zvec = ROL::makeShared<ROL::StdVector<RealT>>(zvec_rcp);
    ROL::SharedPointer<std::vector<RealT> > dvec_rcp = ROL::makeShared<std::vector<RealT>>(nx+2,0.0);
    ROL::SharedPointer<ROL::StdVector<RealT> > dvec = ROL::makeShared<ROL::StdVector<RealT>>(dvec_rcp);
    for ( int i = 0; i < nx+2; i++ ) {
      (*zvec_rcp)[i] = random<RealT>(comm);
      (*dvec_rcp)[i] = random<RealT>(comm);
    }
    ROL::Vector_SimOpt<RealT> x(xu, zvec);
    ROL::Vector_SimOpt<RealT> v(vu, dvec);
    *outStream << std::endl << "TESTING SimulatedConstraint" << std::endl; 
    simcon.checkApplyJacobian(x, v, *vu, true, *outStream);
    simcon.checkAdjointConsistencyJacobian(*vu, v, x, *vu, x, true, *outStream);
    simcon.checkApplyAdjointHessian(x, *vu, v, x, true, *outStream);
    *outStream << std::endl << "TESTING SimulatedObjective" << std::endl;
    simobj.checkGradient(x, v, true, *outStream);
    simobj.checkHessVec(x, v, true, *outStream);
    ROL::Algorithm<RealT> algo("Composite Step", *parlist);
    vu->zero();
    x.scale(1);
    algo.run(x, *vu, simobj, simcon, true, *outStream);

    // Output control to file.
    if (Teuchos::rank<int>(*comm)==0) {
      std::ofstream file;
      file.open("control-fs-expv.txt");
      for ( int i = 0; i < nx+2; ++i ) {
        file << (*zvec_rcp)[i] << "\n";
      }
      file.close();
    }

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
