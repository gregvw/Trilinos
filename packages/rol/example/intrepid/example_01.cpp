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

#include<iostream>

#include "example_01.hpp"

#include "ROL_Vector_SimOpt.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_CompositeStep.hpp"
#include "ROL_ConstraintStatusTest.hpp"

// Minimize (1/2)*||u-utarget||^2+(gamma/2)||z||^2
// 
// subject to the BVP constraint 
//
// -u"+(zu)^2 = f
// u'(0)=u'(L)=0
//
// The target function is x^2*(L-x)^2

using namespace ROL;

typedef double             RealT;
typedef Vector<RealT>      V;
typedef StdVector<RealT>   SV;
typedef std::vector<RealT> vec;


int main(int argc, char *argv[]) {

  
  
     
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  ROL::SharedPointer<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  int errorFlag = 0;
 
  try {

    int numCells = 20;
    int numFields = 2;
    RealT domainLength = 1.0;
    RealT h = domainLength/RealT(numCells);
    RealT gamma = 1e-4;

    int nDoF = numCells*(numFields-1)+1;

    // Create discretization
    ROL::SharedPointer<Discretization<RealT>> disc = ROL::makeShared<Discretization<RealT>>(numCells,numFields,domainLength);
  
    ROL::SharedPointer<vec> u_rcp   = ROL::makeShared<vec>(nDoF,1.0);      // Simulation vector 
    ROL::SharedPointer<vec> z_rcp   = ROL::makeShared<vec>(nDoF,1.0);      // Optimization vector 
    ROL::SharedPointer<vec> yu_rcp  = ROL::makeShared<vec>(nDoF,0.0);      // Test vector in U
    ROL::SharedPointer<vec> yz_rcp  = ROL::makeShared<vec>(nDoF,0.0);      // Test vector in Z

    ROL::SharedPointer<vec> gu_rcp  = ROL::makeShared<vec>(nDoF,0.0);      // Gradient w.r.t. Sim vector
    ROL::SharedPointer<vec> gz_rcp  = ROL::makeShared<vec>(nDoF,0.0);      // Gradient w.r.t. Opt vector

    ROL::SharedPointer<vec> utarget_rcp = ROL::makeShared<vec>(nDoF,1.0);  // Target vector

    ROL::SharedPointer<vec> v_rcp   = ROL::makeShared<vec>(nDoF,1.0);      // Simulation vector 
    ROL::SharedPointer<vec> w_rcp   = ROL::makeShared<vec>(nDoF,1.0);      // Optimization vector 

    ROL::SharedPointer<vec> c_rcp  = ROL::makeShared<vec>(nDoF,0.0);       // Constraint vector
    ROL::SharedPointer<vec> l_rcp  = ROL::makeShared<vec>(nDoF,0.0);       // Lagrange multiplier

    // -----------------------
    // Begin derivative checks
    // -----------------------

    RealT left = -1e0, right = 1e0; 
    for(int i=0;i<nDoF;++i) {

      (*v_rcp)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      (*w_rcp)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      (*yu_rcp)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;
      (*yz_rcp)[i] = ( (RealT)rand() / (RealT)RAND_MAX ) * (right - left) + left;

      RealT x = i*h; // Grid points

      (*utarget_rcp)[i] = x*x*(domainLength-x)*(domainLength-x);
    }

    // Make ROL::StdVector 
    SV u(u_rcp);
    SV z(z_rcp);
    SV gu(u_rcp);
    SV gz(z_rcp);
    SV yu(yu_rcp);
    SV yz(yz_rcp);
    SV v(v_rcp);
    SV w(w_rcp);
    SV c(c_rcp); 
    SV l(l_rcp);

    ROL::SharedPointer<V> utarget = ROL::makeShared<SV>(utarget_rcp); 

    ROL::SharedPointer<V> up   = &u,false;
    ROL::SharedPointer<V> zp   = &z,false;
    ROL::SharedPointer<V> gup  = &gu,false;
    ROL::SharedPointer<V> gzp  = &gz,false;
    ROL::SharedPointer<V> yup  = &yu,false;
    ROL::SharedPointer<V> yzp  = &yz,false;

    Vector_SimOpt<RealT> uz(up,zp);
    Vector_SimOpt<RealT> g(gup,gzp);
    Vector_SimOpt<RealT> y(yup,yzp);

    // Tracking Objective
    ROL::SharedPointer<Objective_SimOpt<RealT>> obj = ROL::makeShared<TrackingObjective<RealT>>(disc,utarget,gamma);

    // Constraint
    ROL::SharedPointer<Constraint_SimOpt<RealT>> con = ROL::makeShared<BVPConstraint<RealT>>(disc);
 
    obj->checkGradient(uz,y,true,*outStream);
    obj->checkHessVec(uz,y,true,*outStream);

    con->checkApplyJacobian(uz,y,c,true,*outStream);
    con->checkApplyAdjointHessian(uz,yz,y,uz,true,*outStream);

    con->checkInverseJacobian_1(c,yu,u,z,true,*outStream);
    con->checkInverseAdjointJacobian_1(c,yu,u,z,true,*outStream);

    con->checkAdjointConsistencyJacobian_1(w,v,u,z,true,*outStream);
    con->checkAdjointConsistencyJacobian_2(w,v,u,z,true,*outStream);
 
    // --------------------
    // End derivative checks 
    // --------------------

  
    // ----------------
    // Run optimization 
    // ----------------  

    // Define algorithm.
    Teuchos::ParameterList parlist;
    std::string stepname = "Composite Step";
    parlist.sublist("Step").sublist(stepname).sublist("Optimality System Solver").set("Nominal Relative Tolerance",1.e-4);
    parlist.sublist("Step").sublist(stepname).sublist("Optimality System Solver").set("Fix Tolerance",true);
    parlist.sublist("Step").sublist(stepname).sublist("Tangential Subproblem Solver").set("Iteration Limit",20);
    parlist.sublist("Step").sublist(stepname).sublist("Tangential Subproblem Solver").set("Relative Tolerance",1e-2);
    parlist.sublist("Step").sublist(stepname).set("Output Level",0);
    parlist.sublist("Status Test").set("Gradient Tolerance",1.e-12);
    parlist.sublist("Status Test").set("Constraint Tolerance",1.e-12);
    parlist.sublist("Status Test").set("Step Tolerance",1.e-14);
    parlist.sublist("Status Test").set("Iteration Limit",100);
    Algorithm<RealT> algo(stepname, parlist);

    // Run algorithm.
    algo.run(uz,g,l,c,*obj,*con,true,*outStream);

  } 
  catch ( std::logic_error err ) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try


  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

