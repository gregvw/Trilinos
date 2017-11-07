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
//         
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

/*! \file  test_07.cpp
    \brief Interior Point test using Hock & Schittkowski problem 32.
*/

#include "ROL_HS32.hpp"
#include "ROL_Algorithm.hpp"


typedef double RealT;

int main(int argc, char *argv[]) {

  
   

  typedef std::vector<RealT>            vec;
  typedef ROL::StdVector<RealT>         SV;
  typedef std::shared_ptr<ROL::Vector<RealT> >      std::shared_ptrV;

//  typedef ROL::PartitionedVector<RealT> PV;


  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream.reset(&std::cout);
  else
    outStream.reset(&bhs);

  int errorFlag = 0;

  try {

    int xopt_dim  = 3; // Dimension of optimization vectors
    int ce_dim = 1;    // Dimension of equality constraint
    int ci_dim = 4;    // Dimension of inequality constraint

    // Exact solution
    std::shared_ptr<vec> x_exact_rcp = std::make_shared<vec>(xopt_dim,0.0);
    (*x_exact_rcp)[xopt_dim-1] = 1.0;

    std::shared_ptr<vec> xopt_rcp = std::make_shared<vec>(xopt_dim,0.0); // Optimization variables

    std::shared_ptr<vec> le_rcp  = std::make_shared<vec>(ce_dim,0.0);    // Equality multiplier
    std::shared_ptr<vec> li_rcp  = std::make_shared<vec>(ci_dim,0.0);    // Inequality multiplier
     
    // Feasible initial guess
    (*xopt_rcp)[0] = 0.1;
    (*xopt_rcp)[1] = 0.7;
    (*xopt_rcp)[2] = 0.2;

    std::shared_ptrV xopt = std::make_shared<SV>(xopt_rcp);
    std::shared_ptrV le  = std::make_shared<SV>(le_rcp);
    std::shared_ptrV li  = std::make_shared<SV>(li_rcp);

    using ROL::ZOO::Objective_HS32;
    using ROL::ZOO::EqualityConstraint_HS32;
    using ROL::ZOO::InequalityConstraint_HS32;    

    std::shared_ptr<ROL::Objective<RealT> > obj_hs32 = std::make_shared<Objective_HS32<RealT>>(); 
    std::shared_ptr<ROL::EqualityConstraint<RealT> > eqcon_hs32 = std::make_shared<EqualityConstraint_HS32<RealT>>();
    std::shared_ptr<ROL::InequalityConstraint<RealT> > incon_hs32 = std::make_shared<InequalityConstraint_HS32<RealT>>();
    
    std::shared_ptr<Teuchos::ParameterList> parlist = std::make_shared<Teuchos::ParameterList>();
    std::string stepname = "Interior Point";

    RealT mu = 0.1;            // Initial penalty parameter
    RealT factor = 0.1;        // Penalty reduction factor

    // Set solver parameters
    parlist->sublist("Step").sublist("Interior Point").set("Initial Barrier Penalty",mu);
    parlist->sublist("Step").sublist("Interior Point").set("Minimium Barrier Penalty",1e-8);
    parlist->sublist("Step").sublist("Interior Point").set("Barrier Penalty Reduction Factor",factor);
    parlist->sublist("Step").sublist("Interior Point").set("Subproblem Iteration Limit",30);

    parlist->sublist("Step").sublist("Composite Step").sublist("Optimality System Solver").set("Nominal Relative Tolerance",1.e-4);
    parlist->sublist("Step").sublist("Composite Step").sublist("Optimality System Solver").set("Fix Tolerance",true);
    parlist->sublist("Step").sublist("Composite Step").sublist("Tangential Subproblem Solver").set("Iteration Limit",20);
    parlist->sublist("Step").sublist("Composite Step").sublist("Tangential Subproblem Solver").set("Relative Tolerance",1e-2);
    parlist->sublist("Step").sublist("Composite Step").set("Output Level",0);

    parlist->sublist("Status Test").set("Gradient Tolerance",1.e-12);
    parlist->sublist("Status Test").set("Constraint Tolerance",1.e-8);
    parlist->sublist("Status Test").set("Step Tolerance",1.e-8);
    parlist->sublist("Status Test").set("Iteration Limit",100);

    ROL::OptimizationProblem<RealT> problem( obj_hs32, xopt, eqcon_hs32, le, incon_hs32, li, parlist);  

    // Define algorithm.
    std::shared_ptr<ROL::Algorithm<RealT> > algo;    
    algo = std::make_shared<ROL::Algorithm<RealT>>(stepname,*parlist);

    algo->run(problem,true,*outStream);   
  
    *outStream << std::endl << std::setw(20) << "Computed Minimizer" << std::setw(20) << "Exact Minimizer" << std::endl;
    for( int i=0;i<xopt_dim;++i ) {   
      *outStream << std::setw(20) << (*xopt_rcp)[i] << std::setw(20) << (*x_exact_rcp)[i] << std::endl;
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

