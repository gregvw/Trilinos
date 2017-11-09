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
    \brief Shows how to solve a finite minimax problem.
*/

#include "ROL_Algorithm.hpp"
#include "ROL_BundleStep.hpp"
#include "ROL_BundleStatusTest.hpp"
#include "ROL_Types.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_Minimax2.hpp"

#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_LAPACK.hpp"

#include <iostream>
#include <algorithm>

typedef double RealT;

int main(int argc, char *argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  // This little trick lets us print to std::cout only if a (dummy) command-line argument is provided.
  int iprint = argc - 1;
  ROL::SharedPointer<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = &std::cout, false;
  else
    outStream = &bhs, false;

  int errorFlag  = 0;

  // *** Example body.

  try {
    // Initialize objective function.
    int dim = 2;
    ROL::Minimax2<RealT> obj;

    // Initialize iteration vectors.
    ROL::SharedPointer<std::vector<RealT> > x_rcp = ROL::makeShared<std::vector<RealT>>(dim, 0.0);
    (*x_rcp)[0] = 1.0; (*x_rcp)[1] = -0.1;
    ROL::StdVector<RealT> x(x_rcp);
    ROL::SharedPointer<std::vector<RealT> > z_rcp = ROL::makeShared<std::vector<RealT>>(dim, 0.0);
    (*z_rcp)[0] = 1.0; (*z_rcp)[1] = 1.0;
    ROL::StdVector<RealT> z(x_rcp);

    // Algorithmic input parameters.
    std::string filename = "input.xml";
    ROL::SharedPointer<Teuchos::ParameterList> parlist = ROL::makeShared<Teuchos::ParameterList>();
    Teuchos::updateParametersFromXmlFile( filename, parlist.ptr() );
    std::string stepname = "Bundle";
    ROL::Algorithm<RealT> algo(stepname,*parlist);

    // Run algorithm.
    algo.run(x, obj, true, *outStream);

    // Compute error 
    ROL::SharedPointer<ROL::Vector<RealT> > diff = x.clone();
    diff->set(x);
    diff->axpy(-1.0,z);
    RealT error = diff->norm();
    *outStream << "\nAbsolute Error: " << error << "\n";
    *outStream <<   "Relative Error: " << error/z.norm() << "\n";
    errorFlag = ((error > 1e2*std::sqrt(ROL::ROL_EPSILON<RealT>())) ? 1 : 0);
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

