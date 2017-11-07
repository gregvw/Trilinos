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

/*! \file  test_02.cpp
    \brief Test creating a LineSearch using an externally provided 
           scalar minimization function. 
*/

#include "ROL_Algorithm.hpp"
#include "ROL_BisectionScalarMinimization.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_Zakharov.hpp"


typedef double RealT;

int main(int argc, char *argv[] ) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

   
  using namespace ROL;

  typedef std::vector<RealT> vector;
  typedef Vector<RealT>      V;
  typedef StdVector<RealT>   SV;

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint     = argc - 1;
  std::shared_ptr<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream.reset(&std::cout);
  else
    outStream.reset(&bhs);

  // Save the format state of the original std::cout.
  Teuchos::oblackholestream oldFormatState;
  oldFormatState.copyfmt(std::cout);

  int errorFlag  = 0;

  // *** Test body.
  try {

    int dim = 10;
   
    std::shared_ptr<vector> x_rcp = std::make_shared<vector>( dim, 0.0 );
    std::shared_ptr<vector> k_rcp = std::make_shared<vector>( dim, 0.0 );

    for (int i=0; i<dim; i++) {
      (*x_rcp)[i]   = 2;
      (*k_rcp)[i]   = i+1.0;
    }

    std::shared_ptr<V> k = std::make_shared<SV>(k_rcp);
    SV x(x_rcp);

    ZOO::Objective_Zakharov<RealT> obj(k);

    Teuchos::ParameterList parlist;

    parlist.sublist("Scalar Minimization").set("Type","Bisection");
    parlist.sublist("Scalar Minimization").sublist("Bisection").set("Tolerance",1.e-10);
    parlist.sublist("Scalar Minimization").sublist("Bisection").set("Iteration Limit",1000);

    std::shared_ptr<ScalarMinimization<RealT> > sm = std::make_shared<BisectionScalarMinimization<RealT>>(parlist);
    std::shared_ptr<LineSearch<RealT> > ls = std::make_shared<ScalarMinimizationLineSearch<RealT>>(parlist, sm);
    std::shared_ptr<Step<RealT> > step = std::make_shared<LineSearchStep<RealT>>( parlist, ls );
   
    std::shared_ptr<StatusTest<RealT> > status = std::make_shared<StatusTest<RealT>>(parlist);

    Algorithm<RealT> algo( step, status, false );
 
    algo.run(x, obj, true, *outStream);  

    RealT abserr = x.norm();

    *outStream << std::scientific << "\n   Absolute Error: " << abserr << std::endl;
    if ( abserr > sqrt(ROL::ROL_EPSILON<RealT>()) ) {
      errorFlag += 1;
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

  // reset format state of std::cout
  std::cout.copyfmt(oldFormatState);


  return 0;
}


