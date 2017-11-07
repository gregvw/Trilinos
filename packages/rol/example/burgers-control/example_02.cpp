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

/*! \file  example_02.cpp
    \brief Shows how to solve a steady Burgers' optimal control problem using
	   the SimOpt interface.  We solve the control problem using Composite
           Step and trust regions.
*/

#include "example_02.hpp"

typedef double RealT;

int main(int argc, char *argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  int errorFlag = 0;

  // *** Example body.

  try {
    // Initialize full objective function.
    int nx      = 256;   // Set spatial discretization.
    RealT alpha = 1.e-3; // Set penalty parameter.
    RealT nu    = 1e-2;  // Viscosity parameter.
    Objective_BurgersControl<RealT> obj(alpha,nx);
    // Initialize equality constraints
    Constraint_BurgersControl<RealT> con(nx,nu);
    Teuchos::ParameterList list;
    list.sublist("SimOpt").sublist("Solve").set("Absolute Residual Tolerance",1.e2*ROL::ROL_EPSILON<RealT>());
    con.setSolveParameters(list);
    // Initialize iteration vectors.
    std::shared_ptr<std::vector<RealT> > z_rcp  = std::make_shared<std::vector<RealT>>(nx+2, 1.0);
    std::shared_ptr<std::vector<RealT> > gz_rcp = std::make_shared<std::vector<RealT>>(nx+2, 1.0);
    std::shared_ptr<std::vector<RealT> > yz_rcp = std::make_shared<std::vector<RealT>>(nx+2, 1.0);
    for (int i=0; i<nx+2; i++) {
      (*z_rcp)[i]  = (RealT)rand()/(RealT)RAND_MAX;
      (*yz_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
    }
    ROL::StdVector<RealT> z(z_rcp);
    ROL::StdVector<RealT> gz(gz_rcp);
    ROL::StdVector<RealT> yz(yz_rcp);
    std::shared_ptr<ROL::Vector<RealT> > zp  = &z,false;
    std::shared_ptr<ROL::Vector<RealT> > gzp = &z,false;
    std::shared_ptr<ROL::Vector<RealT> > yzp = &yz,false;

    std::shared_ptr<std::vector<RealT> > u_rcp  = std::make_shared<std::vector<RealT>>(nx, 1.0);
    std::shared_ptr<std::vector<RealT> > gu_rcp = std::make_shared<std::vector<RealT>>(nx, 1.0);
    std::shared_ptr<std::vector<RealT> > yu_rcp = std::make_shared<std::vector<RealT>>(nx, 1.0);
    for (int i=0; i<nx; i++) {
      (*u_rcp)[i]  = (RealT)rand()/(RealT)RAND_MAX;
      (*yu_rcp)[i] = (RealT)rand()/(RealT)RAND_MAX;
    }
    ROL::StdVector<RealT> u(u_rcp);
    ROL::StdVector<RealT> gu(gu_rcp);
    ROL::StdVector<RealT> yu(yu_rcp);
    std::shared_ptr<ROL::Vector<RealT> > up  = &u,false;
    std::shared_ptr<ROL::Vector<RealT> > gup = &u,false;
    std::shared_ptr<ROL::Vector<RealT> > yup = &yu,false;

    std::shared_ptr<std::vector<RealT> > c_rcp = std::make_shared<std::vector<RealT>>(nx, 1.0);
    std::shared_ptr<std::vector<RealT> > l_rcp = std::make_shared<std::vector<RealT>>(nx, 1.0);
    ROL::StdVector<RealT> c(c_rcp);
    ROL::StdVector<RealT> l(l_rcp);

    ROL::Vector_SimOpt<RealT> x(up,zp);
    ROL::Vector_SimOpt<RealT> g(gup,gzp);
    ROL::Vector_SimOpt<RealT> y(yup,yzp);

    // Check derivatives.
    obj.checkGradient(x,x,y,true,*outStream);
    obj.checkHessVec(x,x,y,true,*outStream);
    con.checkApplyJacobian(x,y,c,true,*outStream);
    con.checkApplyAdjointJacobian(x,yu,c,x,true,*outStream);
    con.checkApplyAdjointHessian(x,yu,y,x,true,*outStream);

    // Initialize reduced objective function.
    std::shared_ptr<std::vector<RealT> > p_rcp  = std::make_shared<std::vector<RealT>>(nx, 1.0);
    ROL::StdVector<RealT> p(p_rcp);
    std::shared_ptr<ROL::Vector<RealT> > pp  = &p,false;
    std::shared_ptr<ROL::Objective_SimOpt<RealT> > pobj = &obj,false;
    std::shared_ptr<ROL::Constraint_SimOpt<RealT> > pcon = &con,false;
    ROL::Reduced_Objective_SimOpt<RealT> robj(pobj,pcon,up,zp,pp);
    // Check derivatives.
    robj.checkGradient(z,z,yz,true,*outStream);
    robj.checkHessVec(z,z,yz,true,*outStream);

    // Get parameter list.
    std::string filename = "input.xml";
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    parlist->sublist("Status Test").set("Gradient Tolerance",1.e-14);
    parlist->sublist("Status Test").set("Constraint Tolerance",1.e-14);
    parlist->sublist("Status Test").set("Step Tolerance",1.e-16);
    parlist->sublist("Status Test").set("Iteration Limit",1000);
    // Declare ROL algorithm pointer.
    std::shared_ptr<ROL::Algorithm<RealT> > algo;

    // Run optimization with Composite Step.
    algo = std::make_shared<ROL::Algorithm<RealT>>("Composite Step",*parlist,false);
    RealT zerotol = std::sqrt(ROL::ROL_EPSILON<RealT>());
    z.zero();
    con.solve(c,u,z,zerotol);
    c.zero(); l.zero();
    algo->run(x, g, l, c, obj, con, true, *outStream);
    std::shared_ptr<ROL::Vector<RealT> > zCS = z.clone();
    zCS->set(z);

    // Run Optimization with Trust-Region algorithm.
    algo = std::make_shared<ROL::Algorithm<RealT>>("Trust Region",*parlist,false);
    z.zero();
    algo->run(z,robj,true,*outStream);

    // Check solutions.
    std::shared_ptr<ROL::Vector<RealT> > err = z.clone();
    err->set(*zCS); err->axpy(-1.,z);
    errorFlag += ((err->norm()) > 1.e-8) ? 1 : 0;
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

