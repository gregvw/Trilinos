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
Real random(const std::shared_ptr<const Teuchos::Comm<int> > &comm) {
  Real val = 0.0;
  if ( Teuchos::rank<int>(*comm)==0 ) {
    val = (Real)rand()/(Real)RAND_MAX;
  }
  Teuchos::broadcast<int,Real>(*comm,0,1,&val);
  return val;
}

int main(int argc, char* argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  std::shared_ptr<const Teuchos::Comm<int> > comm
    = Teuchos::DefaultComm<int>::getComm();

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  std::shared_ptr<std::ostream> outStream;
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
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    /**********************************************************************************************/
    /************************* CONSTRUCT VECTORS **************************************************/
    /**********************************************************************************************/
    // Build control vectors
    int nx = 256;
    std::shared_ptr<std::vector<RealT> > x1_rcp  = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> x1(x1_rcp);
    std::shared_ptr<std::vector<RealT> > x2_rcp  = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> x2(x2_rcp);
    std::shared_ptr<std::vector<RealT> > x3_rcp  = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> x3(x3_rcp);
    std::shared_ptr<std::vector<RealT> > z_rcp  = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> z(z_rcp);
    std::shared_ptr<std::vector<RealT> > xr_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> xr(xr_rcp);
    std::shared_ptr<std::vector<RealT> > d_rcp  = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    ROL::StdVector<RealT> d(d_rcp);
    for ( int i = 0; i < nx+2; i++ ) {
      (*xr_rcp)[i] = random<RealT>(comm);
      (*d_rcp)[i]  = random<RealT>(comm);
    }
    // Build state and adjoint vectors
    std::shared_ptr<std::vector<RealT> > u_rcp  = std::make_shared<std::vector<RealT>>(nx,1.0);
    ROL::StdVector<RealT> u(u_rcp);
    std::shared_ptr<std::vector<RealT> > p_rcp  = std::make_shared<std::vector<RealT>>(nx,0.0);
    ROL::StdVector<RealT> p(p_rcp);
    std::shared_ptr<ROL::Vector<RealT> > up = &u,false;
    std::shared_ptr<ROL::Vector<RealT> > zp = &z,false;
    std::shared_ptr<ROL::Vector<RealT> > pp = &p,false;
    /**********************************************************************************************/
    /************************* CONSTRUCT SOL COMPONENTS *******************************************/
    /**********************************************************************************************/
    // Build samplers
    int dim = 4;
    int nSamp = parlist->sublist("Problem Description").get("Number of Samples", 20);
    std::vector<RealT> tmp(2,0.0); tmp[0] = -1.0; tmp[1] = 1.0;
    std::vector<std::vector<RealT> > bounds(dim,tmp);
    std::shared_ptr<ROL::BatchManager<RealT> > bman
      = std::make_shared<ROL::StdTeuchosBatchManager<RealT,int>>(comm);
    std::shared_ptr<ROL::SampleGenerator<RealT> > sampler
      = std::make_shared<ROL::MonteCarloGenerator<RealT>>(nSamp,bounds,bman,false,false,100);
    int nprocs = Teuchos::size<int>(*comm);
    std::stringstream name;
    name << "samples_np" << nprocs;
    sampler->print(name.str());
    /**********************************************************************************************/
    /************************* CONSTRUCT OBJECTIVE FUNCTION ***************************************/
    /**********************************************************************************************/
    // Build risk-averse objective function
    RealT alpha = 1.e-3;
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > pobjSimOpt
      = std::make_shared<Objective_BurgersControl<RealT>>(alpha,nx);
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > pconSimOpt
      = std::make_shared<Constraint_BurgersControl<RealT>>(nx);
    std::shared_ptr<ROL::Objective<RealT> > pObj
      = std::make_shared<ROL::Reduced_Objective_SimOpt<RealT>>(pobjSimOpt,pconSimOpt,up,zp,pp);
    std::shared_ptr<ROL::Objective<RealT> > obj = std::make_shared<ROL::RiskNeutralObjective<RealT>>(pObj, sampler, true);
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
    int nvecloc = sampler->numMySamples();
    std::vector<std::shared_ptr<std::vector<RealT> > >  xuk_nvecloc),  vuk_rcp(nvecloc),  yuk_rcp(nvecloc;
    std::vector<std::shared_ptr<std::vector<RealT> > > dxuk_nvecloc), dvuk_rcp(nvecloc), dyuk_rcp(nvecloc;
    std::vector<std::shared_ptr<ROL::Vector<RealT> > >  xu_nvecloc),   vu_rcp(nvecloc),   yu_rcp(nvecloc;
    std::vector<std::shared_ptr<ROL::Vector<RealT> > > dxu_nvecloc),  dvu_rcp(nvecloc),  dyu_rcp(nvecloc;
    RealT right = 1, left = 0;
    for( int k=0; k<nvecloc; ++k ) {
      xuk_rcp[k] = std::make_shared<std::vector<RealT>>(nx,1.0);
      vuk_rcp[k] = std::make_shared<std::vector<RealT>>(nx,1.0);
      yuk_rcp[k] = std::make_shared<std::vector<RealT>>(nx,1.0);
      dxuk_rcp[k] = std::make_shared<std::vector<RealT>>(nx,1.0);
      dvuk_rcp[k] = std::make_shared<std::vector<RealT>>(nx,1.0);
      dyuk_rcp[k] = std::make_shared<std::vector<RealT>>(nx,1.0);
      xu_rcp[k] = std::make_shared<ROL::StdVector<RealT>>( xuk_rcp[k] );
      vu_rcp[k] = std::make_shared<ROL::StdVector<RealT>>( vuk_rcp[k] );
      yu_rcp[k] = std::make_shared<ROL::StdVector<RealT>>( yuk_rcp[k] );
      dxu_rcp[k] = std::make_shared<ROL::StdVector<RealT>>( dxuk_rcp[k] );
      dvu_rcp[k] = std::make_shared<ROL::StdVector<RealT>>( dvuk_rcp[k] );
      dyu_rcp[k] = std::make_shared<ROL::StdVector<RealT>>( dyuk_rcp[k] );
      for( int i=0; i<nx; ++i ) {
        (*xuk_rcp[k])[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
        (*vuk_rcp[k])[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
        (*yuk_rcp[k])[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
        (*dxuk_rcp[k])[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
        (*dvuk_rcp[k])[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
        (*dyuk_rcp[k])[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      }
    }
    std::shared_ptr<ROL::PrimalSimulatedVector<RealT> > xu, vu, yu;
    std::shared_ptr<ROL::DualSimulatedVector<RealT> > dxu, dvu, dyu;
    xu = std::make_shared<ROL::PrimalSimulatedVector<RealT>>(xu_rcp, bman, sampler);
    vu = std::make_shared<ROL::PrimalSimulatedVector<RealT>>(vu_rcp, bman, sampler);
    yu = std::make_shared<ROL::PrimalSimulatedVector<RealT>>(yu_rcp, bman, sampler);
    dxu = std::make_shared<ROL::DualSimulatedVector<RealT>>(dxu_rcp, bman, sampler);
    dvu = std::make_shared<ROL::DualSimulatedVector<RealT>>(dvu_rcp, bman, sampler);
    dyu = std::make_shared<ROL::DualSimulatedVector<RealT>>(dyu_rcp, bman, sampler);
    // SimOpt vectors.
    std::shared_ptr<std::vector<RealT> > xz_rcp, vz_rcp, yz_rcp;
    std::shared_ptr<std::vector<RealT> > dxz_rcp, dvz_rcp, dyz_rcp;
    xz_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    vz_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    yz_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    dxz_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    dvz_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    dyz_rcp = std::make_shared<std::vector<RealT>>(nx+2,0.0);
    std::shared_ptr<ROL::StdVector<RealT> > xz, vz, yz;
    std::shared_ptr<ROL::StdVector<RealT> > dxz, dvz, dyz;
    xz = std::make_shared<ROL::StdVector<RealT>>(xz_rcp);
    vz = std::make_shared<ROL::StdVector<RealT>>(vz_rcp);
    yz = std::make_shared<ROL::StdVector<RealT>>(yz_rcp);
    dxz = std::make_shared<ROL::StdVector<RealT>>(dxz_rcp);
    dvz = std::make_shared<ROL::StdVector<RealT>>(dvz_rcp);
    dyz = std::make_shared<ROL::StdVector<RealT>>(dyz_rcp);
    for ( int i = 0; i < nx+2; i++ ) {
      (*xz_rcp)[i] = random<RealT>(comm);
      (*vz_rcp)[i] = random<RealT>(comm);
      (*yz_rcp)[i] = random<RealT>(comm);
      (*dxz_rcp)[i] = random<RealT>(comm);
      (*dvz_rcp)[i] = random<RealT>(comm);
      (*dyz_rcp)[i] = random<RealT>(comm);
    }
    ROL::Vector_SimOpt<RealT> x(xu, xz);
    ROL::Vector_SimOpt<RealT> v(vu, vz);
    ROL::Vector_SimOpt<RealT> y(yu, yz);
    ROL::Vector_SimOpt<RealT> dx(dxu, dxz);
    ROL::Vector_SimOpt<RealT> dv(dvu, dvz);
    ROL::Vector_SimOpt<RealT> dy(dyu, dyz);
    xu->checkVector(*vu,*yu,true,*outStream);
    x.checkVector(v,y,true,*outStream);
    dxu->checkVector(*dvu,*dyu,true,*outStream);
    dx.checkVector(dv,dy,true,*outStream);
    *outStream << std::endl << "TESTING SimulatedConstraint" << std::endl; 
    simcon.checkApplyJacobian(x, v, *dvu, true, *outStream);
    simcon.checkAdjointConsistencyJacobian(*vu, v, x, true, *outStream);
    simcon.checkApplyAdjointHessian(x, *vu, v, dx, true, *outStream);
    *outStream << std::endl << "TESTING SimulatedObjective" << std::endl;
    simobj.checkGradient(x, v, true, *outStream);
    simobj.checkHessVec(x, v, true, *outStream);
    ROL::Algorithm<RealT> algo("Composite Step", *parlist);
    vu->zero();
    algo.run(x, *vu, simobj, simcon, true, *outStream);

    // Output control to file.
    if (Teuchos::rank<int>(*comm)==0) {
      std::ofstream file;
      file.open("control-fs-expv.txt");
      for ( int i = 0; i < nx+2; ++i ) {
        file << (*xz_rcp)[i] << "\n";
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
